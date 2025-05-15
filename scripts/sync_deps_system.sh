#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath $0)")

uv pip install setuptools --system
uv pip install torch --index-url https://download.pytorch.org/whl/cu118 --system
uv pip install --system -r "$SCRIPT_DIR/../pyproject.toml"
