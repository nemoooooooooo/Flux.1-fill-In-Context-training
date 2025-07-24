#!/usr/bin/env bash
set -euo pipefail

# Activate the project virtual environment
source /workspace/FLUX.1-Kontext-dev-Training/.venv/bin/activate

#############################
# Cache & environment setup #
#############################

export HF_HOME=/workspace/cache/huggingface
export TRANSFORMERS_CACHE=/workspace/cache/huggingface/hub
export TORCH_HOME=/workspace/cache/torch
export PIP_CACHE_DIR=/workspace/cache/pip

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR"

# (Optional) If you want to guarantee code uses an explicit path:
export HF_CACHE_DIR="$HF_HOME"




#############################
# Launch training           #
#############################
accelerate launch train.py 

