#!/bin/bash
ulimit -u 8192

DEFAULT_RUN_NAME="ocr-qwen2-vl-7b-align"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=512
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

STAGE_PATH=${1:-"Efficient-Large-Model/Qwen2-VL-7B-Instruct"}
DATA_MIXTURE=${2:-"llava_15_mix"}
OUTPUT_DIR=${3:-"runs/train/ocr-qwen2-vl-8b-align"}

source scripts/setups/train.sh

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE_PATH \
        --chat_template qwen2 \
        --data_mixture $DATA_MIXTURE \
        --vision_tower /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/checkpoints/sam_clip_ckpt/model_cache/model-00001-of-000001.safetensors \
        --mm_vision_select_feature cls_patch \
        --mm_projector linear \
        --tune_vision_tower False \
        --tune_mm_projector True \
        --tune_language_model False \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --report_to wandb 


# #!/bin/bash

# DEFAULT_RUN_NAME="ocr-qwen2-vl-7b-align"
# DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=512
# DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

# STAGE_PATH=${1:-"Efficient-Large-Model/Qwen2-VL-7B-Instruct"}
# DATA_MIXTURE=${2:-"llava_15_mix"}
# OUTPUT_DIR=${3:-"runs/train/-8b-align"}

# source scripts/setups/train.sh

# # Set memory-efficient environment variables
# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export TOKENIZERS_PARALLELISM=false

# torchrun \
#     --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
#     llava/train/train_mem.py \
#         --deepspeed scripts/zero3.json \
#         --model_name_or_path $STAGE_PATH \
#         --chat_template qwen2 \
#         --data_mixture $DATA_MIXTURE \
#         --vision_tower /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/checkpoints/sam_clip_ckpt/model_cache/model-00001-of-000001.safetensors \
#         --mm_vision_select_feature cls_patch \
#         --mm_projector linear \
#         --tune_vision_tower False \
#         --tune_mm_projector True \
#         --tune_language_model False \
#         --mm_vision_select_layer -2 \
#         --mm_use_im_start_end False \
#         --mm_use_im_patch_token False \
#         --image_aspect_ratio dynamic \
#         --bf16 True \
#         --output_dir $OUTPUT_DIR/model \
#         --num_train_epochs 1 \
#         --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
#         --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#         --evaluation_strategy no \
#         --save_strategy steps \
#         --save_steps 100 \
#         --save_total_limit 1 \
#         --learning_rate 1e-3 \
#         --weight_decay 0. \
#         --warmup_ratio 0.03 \
#         --lr_scheduler_type cosine \
#         --logging_steps 1 \
#         --model_max_length 4096 \
#         --gradient_checkpointing True \
#         --dataloader_num_workers 4 \
#         --dataloader_pin_memory False \
#         --dataloader_prefetch_factor 2 \
#         --report_to wandb