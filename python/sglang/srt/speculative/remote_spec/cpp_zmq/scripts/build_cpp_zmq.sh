#!/bin/bash
set -euxo pipefail

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
pushd "$PROJECT_DIR"

# Install dependencies if needed
pip install pybind11 msgpack --break-system-packages
apt-get update && apt-get install -y libmsgpack-dev libzmq3-dev

# Build the extension
python3 setup.py build_ext --inplace
popd
