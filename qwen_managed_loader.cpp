#include <torch/extension.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <unordered_map>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::ostringstream oss;                                               \
      oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "      \
          << cudaGetErrorString(err);                                      \
      throw std::runtime_error(oss.str());                                  \
    }                                                                       \
  } while (0)

struct MmapInfo {
  void* host_ptr;
  void* device_ptr;
  size_t size;

  std::atomic<bool> released{false};

  MmapInfo(void* h, void* d, size_t s) : host_ptr(h), device_ptr(d), size(s) {}

  bool try_mark_released() {
    bool expected = false;
    return released.compare_exchange_strong(expected, true);
  }

  void cleanup() {
    if (!try_mark_released()) {
      return;
    }
    if (host_ptr && host_ptr != MAP_FAILED) {
      cudaHostUnregister(host_ptr);  // unpin (logical release)
      munmap(host_ptr, size);         // unmap from process address space
    }
    // The mapping is released. Because the payload is file-backed, Linux may
    // keep clean pages in the page cache and satisfy later mmaps without SSD I/O.
  }
};

static std::unordered_map<void*, std::shared_ptr<MmapInfo>> g_mmap_registry;
static std::mutex g_mmap_mutex;

static void erase_mapping(void* device_ptr) {
  std::lock_guard<std::mutex> lock(g_mmap_mutex);
  g_mmap_registry.erase(device_ptr);
}

static void CUDART_CB release_mmap_callback(void* user_data) {
  auto* holder = static_cast<std::shared_ptr<MmapInfo>*>(user_data);
  if (holder && *holder) {
    (*holder)->cleanup();
  }
  if (holder && *holder) {
    erase_mapping((*holder)->device_ptr);
  }
  delete holder;
}

static bool env_true(const char* name) {
  const char* v = std::getenv(name);
  if (!v) return false;
  return v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T';
}

torch::Tensor load_bin_to_managed_1d_mmap(const std::string& filename,
                                          int64_t num_elements) {
  if (num_elements <= 0) {
    throw std::runtime_error("num_elements must be > 0");
  }

  const size_t elem_size = 2;  // float16
  const size_t expected_size = static_cast<size_t>(num_elements) * elem_size;

  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd);
    throw std::runtime_error("Could not stat file: " + filename);
  }
  if (static_cast<size_t>(st.st_size) != expected_size) {
    close(fd);
    throw std::runtime_error(
        "File size mismatch for " + filename +
        ", got " + std::to_string(st.st_size) +
        ", expected " + std::to_string(expected_size));
  }

  int mmap_flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
  if (env_true("QWEN_MMAP_POPULATE")) {
    mmap_flags |= MAP_POPULATE;
  }
#endif

  if (env_true("QWEN_POSIX_FADVISE_SEQ")) {
    (void)posix_fadvise(fd, 0, static_cast<off_t>(expected_size), POSIX_FADV_SEQUENTIAL);
  }

  void* mapped_ptr = mmap(nullptr, expected_size, PROT_READ, mmap_flags, fd, 0);
  close(fd);

  if (mapped_ptr == MAP_FAILED) {
    throw std::runtime_error("mmap failed for file: " + filename);
  }

  if (env_true("QWEN_MADVISE_WILLNEED")) {
    (void)madvise(mapped_ptr, expected_size, MADV_WILLNEED);
  }

  if (env_true("QWEN_PREFETCH_PAGES")) {
    const long page = sysconf(_SC_PAGESIZE);
    if (page > 0) {
      volatile const char* p = static_cast<const char*>(mapped_ptr);
      for (size_t off = 0; off < expected_size; off += static_cast<size_t>(page)) {
        (void)p[off];
      }
    }
  }

  cudaError_t err = cudaHostRegister(mapped_ptr, expected_size, 
                                     cudaHostRegisterReadOnly | 
                                     cudaHostRegisterMapped);
  if (err != cudaSuccess) {
    munmap(mapped_ptr, expected_size);
    throw std::runtime_error("cudaHostRegister failed: " + 
                            std::string(cudaGetErrorString(err)));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, mapped_ptr, 0);
  if (err != cudaSuccess) {
    cudaHostUnregister(mapped_ptr);
    munmap(mapped_ptr, expected_size);
    throw std::runtime_error("cudaHostGetDevicePointer failed: " + 
                            std::string(cudaGetErrorString(err)));
  }

  auto* mmap_info = new MmapInfo(mapped_ptr, device_ptr, expected_size);
  auto mmap_shared = std::shared_ptr<MmapInfo>(mmap_info);
  {
    std::lock_guard<std::mutex> lock(g_mmap_mutex);
    g_mmap_registry[device_ptr] = mmap_shared;
  }

  auto options = torch::TensorOptions()
                     .dtype(torch::kHalf)
                     .device(torch::kCUDA);

  auto deleter = [mmap_shared](void* p) {
    mmap_shared->cleanup();
    erase_mapping(mmap_shared->device_ptr);
  };

  torch::Tensor tensor = torch::from_blob(
      device_ptr,
      {num_elements},
      /*stride=*/{1},
      deleter,
      options);

  return tensor;
}

void release_mmap_async(torch::Tensor tensor) {
  if (!tensor.defined()) {
    return;
  }
  void* device_ptr = tensor.data_ptr();
  std::shared_ptr<MmapInfo> mmap_shared;
  {
    std::lock_guard<std::mutex> lock(g_mmap_mutex);
    auto it = g_mmap_registry.find(device_ptr);
    if (it == g_mmap_registry.end()) {
      return;
    }
    mmap_shared = it->second;
    g_mmap_registry.erase(it);
  }
  auto* holder = new std::shared_ptr<MmapInfo>(mmap_shared);
  auto stream = at::cuda::getDefaultCUDAStream();
  CHECK_CUDA(cudaLaunchHostFunc(stream.stream(), release_mmap_callback, holder));
}

void release_mmap_async_stream(torch::Tensor tensor, uint64_t stream_ptr) {
  if (!tensor.defined()) {
    return;
  }
  void* device_ptr = tensor.data_ptr();
  std::shared_ptr<MmapInfo> mmap_shared;
  {
    std::lock_guard<std::mutex> lock(g_mmap_mutex);
    auto it = g_mmap_registry.find(device_ptr);
    if (it == g_mmap_registry.end()) {
      return;
    }
    mmap_shared = it->second;
    g_mmap_registry.erase(it);
  }
  auto* holder = new std::shared_ptr<MmapInfo>(mmap_shared);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  CHECK_CUDA(cudaLaunchHostFunc(stream, release_mmap_callback, holder));
}

void release_mmap_sync(torch::Tensor tensor) {
  if (!tensor.defined()) {
    return;
  }
  void* device_ptr = tensor.data_ptr();
  std::shared_ptr<MmapInfo> mmap_shared;
  {
    std::lock_guard<std::mutex> lock(g_mmap_mutex);
    auto it = g_mmap_registry.find(device_ptr);
    if (it == g_mmap_registry.end()) {
      return;
    }
    mmap_shared = it->second;
    g_mmap_registry.erase(it);
  }
  if (mmap_shared) {
    mmap_shared->cleanup();
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_bin_to_managed_1d_mmap",
        &load_bin_to_managed_1d_mmap,
        "Load float16 .bin using mmap (zero-copy for unified memory)");
  m.def("release_mmap_async",
        &release_mmap_async,
        "Release mmap+cudaHostRegister asynchronously on default stream");
  m.def("release_mmap_async_stream",
        &release_mmap_async_stream,
        "Release mmap+cudaHostRegister asynchronously on given stream");
  m.def("release_mmap_sync",
        &release_mmap_sync,
        "Release mmap+cudaHostRegister synchronously");
}
