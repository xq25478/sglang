#!/bin/bash
set -ex && clear

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd "$SCRIPT_DIR"

# 安装依赖（如果需要）
pip config set global.index-url https://mirrors.jd.com/pypi/web/simple
pip config set global.trusted-host mirrors.jd.com
pip install pybind11 msgpack

# 清理旧 build
rm -rf build *.so

# 编译扩展
python3 setup.py build_ext --inplace

# export REMOTE_SPEC_DEBUG=1
# 测试（可选）
# python3 test_remote_spec_zmq.py

popd