#!/bin/bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
pushd "$PROJECT_DIR"

pip install pybind11 msgpack --break-system-packages
apt-get update && apt-get install -y libmsgpack-dev libzmq3-dev

python3 setup.py build_ext --inplace
popd
