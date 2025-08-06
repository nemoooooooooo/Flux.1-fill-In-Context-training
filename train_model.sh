#!/usr/bin/env bash
set -euo pipefail

# Activate the project virtual environment
source /workspace/Flux_fill_Incontext/.venv/bin/activate

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


# Define variables for columns
export SOURCE_COLUMN="blur"
export TARGET_COLUMN="target"
export MASK_COLUMN="mask"
export CAPTION_COLUMN="gender"
export MODEL_NAME="/workspace/cache/huggingface/models--black-forest-labs--FLUX.1-Fill-dev/snapshots/358293da0354175698b67ec8299acf928313a78a"
export TRAIN_DATASET_NAME="SnapwearAI/SAKS_INCONTEXT_Transformations"
export TEST_DATASET_NAME="SnapwearAI/SAKS_INCONTEXT_Transformations"
# Make output path explicit under /workspace
export PROJECT_ROOT="/workspace/Flux_fill_Incontext"
export OUTPUT_DIR="${PROJECT_ROOT}/blur_SAKS_lora_weights"   # in v1 the source was masked target. change it to ghost.

mkdir -p "$OUTPUT_DIR"

LOG_ROOT=/workspace/debug_logs

#############################
# Launch training           #
#############################
accelerate launch train.py \
    --pretrained_model_name_or_path="$MODEL_NAME" \
    --dataset_name="$TRAIN_DATASET_NAME" \
    --source_image_column="$SOURCE_COLUMN" \
    --target_image_column="$TARGET_COLUMN" \
    --mask_column="$MASK_COLUMN" \
    --caption_column="$CAPTION_COLUMN" \
    --caption_template "Two-panel image showcase image transformation from a blurred image to realistic 4K photograph;[IMAGE1] blurred photograph.[IMAGE2] Lifelike studio DSLR photograph of the {cap} model, natural skin texture, fashion photography aesthetic" \
    --aspect_ratio_buckets="768,576" \
    --random_flip \
    --output_dir="$OUTPUT_DIR" \
    --validation_steps=400 \
    --checkpointing_steps=500 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-5 \
    --mixed_precision=bf16 \
    --tracker_project_name="SAKS_conditional_generation" \
    --train_mode="lora" \