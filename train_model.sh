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
    --caption_template "Two-panel image showcase image transformation from stick figure drawing to realistic 4K photograph;[IMAGE1] stick figure.[IMAGE2] Lifelike studio DSLR photograph of the {cap} model, natural skin texture, fashion photography aesthetic" \
    --aspect_ratio_buckets="768,576" \
    --random_flip \
    --output_dir="$OUTPUT_DIR" \
    --validation_steps=1000 \
    --checkpointing_steps=2000 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-5 \
    --mixed_precision=bf16 \
    --tracker_project_name="SAKS_conditional_generation" \
    