import re
import subprocess
import threading
import time
from typing import List, Optional, Tuple


class JetsonMemoryProfiler:
    def __init__(self, profiling_type: str = "model", interval: int = 100, verbose: bool = False):
        self.profiling_type = profiling_type
        self.interval = interval
        self.verbose = verbose
        self.timestamps: List[float] = []
        self.memory_cum_delta: List[int] = []
        self.initial_mem: Optional[int] = None
        self.start_time: Optional[float] = None
        self.stop_event = threading.Event()
        self.data_thread: Optional[threading.Thread] = None
        self.tegrastats_process: Optional[subprocess.Popen] = None
        self.ram_pattern = re.compile(r"RAM (\d+)/(\d+)MB")

    def _capture_data(self) -> None:
        while not self.stop_event.is_set():
            if not self.tegrastats_process or self.tegrastats_process.poll() is not None:
                break

            line = self.tegrastats_process.stdout.readline()
            if not line:
                break

            match = self.ram_pattern.search(line)
            if not match:
                continue

            current_used = int(match.group(1))
            if self.initial_mem is None:
                self.initial_mem = current_used
                self.start_time = time.time()
                if self.verbose:
                    print(f"[Memory] baseline: {self.initial_mem} MB")
                continue

            if self.start_time is None:
                continue
            self.timestamps.append(time.time() - self.start_time)
            self.memory_cum_delta.append(current_used - self.initial_mem)

    def start_profiling(self) -> None:
        if self.verbose:
            print(
                f"--- Starting Jetson memory profiling "
                f"(type={self.profiling_type}, interval={self.interval}ms) ---"
            )
        self.tegrastats_process = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.stop_event.clear()
        self.data_thread = threading.Thread(target=self._capture_data, daemon=True)
        self.data_thread.start()
        time.sleep(0.2)

    def stop_profiling(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        if self.tegrastats_process and self.tegrastats_process.poll() is None:
            self.tegrastats_process.terminate()
            try:
                self.tegrastats_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.tegrastats_process.kill()
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=1)
        self.tegrastats_process = None
        self.data_thread = None

    def global_memory_stats(self) -> Tuple[float, float]:
        if not self.memory_cum_delta or not self.timestamps:
            return 0.0, 0.0

        global_peak = max(self.memory_cum_delta)
        global_average = sum(self.memory_cum_delta) / len(self.memory_cum_delta)
        return global_peak, global_average

    def print_statistics(self, generated_tokens: int, elapsed_seconds: float) -> None:
        if not self.memory_cum_delta or not self.timestamps:
            print("No memory profiling data collected.")
            return

        global_peak, global_average = self.global_memory_stats()
        peak_gb = global_peak / 1024.0
        tokens_per_minute = 0.0
        if elapsed_seconds > 0:
            tokens_per_minute = generated_tokens / elapsed_seconds * 60.0
        throughput_efficiency = 0.0
        if peak_gb > 0:
            throughput_efficiency = tokens_per_minute / peak_gb

        print("\n" + "=" * 60)
        print(f"          NVIDIA Jetson Memory Statistics (type: {self.profiling_type})")
        print("=" * 60)
        print(f"Peak memory increase: {global_peak:.2f} MB (relative to baseline)")
        print(f"Average memory increase: {global_average:.2f} MB (relative to baseline)")
        print(f"Throughput efficiency: {throughput_efficiency:.2f} tok/min/GB")
        print("=" * 60 + "\n")
