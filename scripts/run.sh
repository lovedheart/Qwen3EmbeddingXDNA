#!/bin/bash
export PATH="$HOME/.local/bin:$HOME/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/bin:$PATH"
source ~/projects/mlir-aie/ironenv314/bin/activate
source ~/projects/mlir-aie/utils/env_setup.sh >/dev/null 2>&1
script=$(realpath "$1"); shift
cd "$(dirname "$script")" && exec python "$(basename "$script")" "$@"
