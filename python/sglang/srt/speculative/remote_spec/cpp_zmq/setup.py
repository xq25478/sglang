# setup.py
import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

# ==============================
# zmq 库路径
# ==============================
zmq_include_dir = "./"       # 根据你系统 zmq.hpp 所在路径修改
zmq_lib_dir = "/usr/lib"               # zmq 库路径
zmq_libs = ["zmq"]                     # 链接 libzmq.so

# ==============================
# 扩展模块
# ==============================
ext_modules = [
    Extension(
        "remote_spec_zmq",  # 编译出来的 Python 模块名
        sources=[
            "remote_spec_zmq.cpp",
            "remote_spec_zmq_logging.cpp",
            "remote_spec_zmq_serialization.cpp",
            "remote_spec_zmq_endpoints.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),      # pybind11 头文件
            zmq_include_dir,
        ],
        library_dirs=[zmq_lib_dir],
        libraries=zmq_libs,
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-Wall", "-fPIC"],
        extra_link_args=[],
    )
]

# ==============================
# setup
# ==============================
setup(
    name="remote_spec_zmq",
    version="0.1.0",
    author="zhangyu",
    description="Full duplex ZMQ C++ module for Python",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
