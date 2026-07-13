#!/bin/bash
# Source this script to put CUDA forward-compat driver libraries before
# host-injected libcuda.so when running CUDA 13 userspace on older drivers.

sglang_ci_cuda_compat_contains_path() {
    local path_list=$1
    local item=$2
    [[ ":${path_list}:" == *":${item}:"* ]]
}

sglang_ci_setup_cuda_compat() {
    local compat_path=""
    local compat_dir

    for compat_dir in /usr/local/cuda-13.0/compat /usr/local/cuda/compat /usr/local/cuda-*/compat; do
        if [[ ! -d "${compat_dir}" ]]; then
            continue
        fi
        if sglang_ci_cuda_compat_contains_path "${compat_path}" "${compat_dir}"; then
            continue
        fi

        if [[ -n "${compat_path}" ]]; then
            compat_path="${compat_path}:${compat_dir}"
        else
            compat_path="${compat_dir}"
        fi
    done

    if [[ -n "${compat_path}" ]]; then
        if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
            export LD_LIBRARY_PATH="${compat_path}:${LD_LIBRARY_PATH}"
        else
            export LD_LIBRARY_PATH="${compat_path}"
        fi
        echo "[SGLang CI] CUDA compat LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    else
        echo "[SGLang CI] WARN: 未找到 CUDA compat 目录，CUDA 13 可能要求宿主机 driver >= 580"
    fi
}

sglang_ci_print_cuda_driver_probe() {
    python3 - <<'PYCUDA'
import ctypes

try:
    import torch
except Exception as exc:
    print("[SGLang CI] WARN: failed to import torch for CUDA probe:", exc)
    raise SystemExit(0)

print("[SGLang CI] torch version:", torch.__version__)
print("[SGLang CI] torch cuda:", torch.version.cuda)

for lib_name in ("libcuda.so.1", "libcuda.so"):
    try:
        libcuda = ctypes.CDLL(lib_name)
        break
    except OSError:
        libcuda = None
else:
    libcuda = None

if libcuda is None:
    print("[SGLang CI] WARN: failed to load libcuda.so for CUDA driver probe")
else:
    version = ctypes.c_int()
    ret = libcuda.cuDriverGetVersion(ctypes.byref(version))
    if ret == 0:
        print("[SGLang CI] CUDA driver API version:", version.value)
    else:
        print("[SGLang CI] WARN: cuDriverGetVersion failed:", ret)
PYCUDA
}

sglang_ci_setup_cuda_compat

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    sglang_ci_print_cuda_driver_probe
fi
