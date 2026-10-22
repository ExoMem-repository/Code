from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="qwen_managed_loader",
    ext_modules=[
        CUDAExtension(
            name="qwen_managed_loader",
            sources=[os.path.join(this_dir, "qwen_managed_loader.cpp")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)