#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath $0)")

if [ ! -d "$SCRIPT_DIR/../.venv" ]; then
  echo "Creating venv as does not yet exist"
  uv venv
fi

uv pip install setuptools numpy
#NOTE: will have to run this when 129 is not current versions uv pip install torch --index-url https://download.pytorch.org/whl/cu129
uv pip install torch
uv sync
