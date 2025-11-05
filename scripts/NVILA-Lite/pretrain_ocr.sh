

#!/bin/bash
# ulimit -u 8192
DEFAULT_RUN_NAME="ocr-qwen2-vl-7b-pretrain-debug"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=32
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

STAGE_PATH=${1:-"runs/train/ocr-qwen2-vl-8b-align/model"}
DATA_MIXTURE=${2:-"olmOCR-mix-pretrain"}
OUTPUT_DIR=${3:-"runs/train/ocr-qwen2-vl-8b-pretrain-debug"}

source scripts/setups/train.sh
export WANDB_MODE=offline
export WANDB_SERVICE_WAIT=1200
export WANDB_INIT_TIMEOUT=1200

GPUS_PER_NODE=2
CUDA_VISIBLE_DEVICES=2,3
MASTER_PORT=25003

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3_new.json \
        --model_name_or_path $STAGE_PATH \
        --data_mixture $DATA_MIXTURE \
        --vision_tower /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/checkpoints/sam_clip_ckpt/model_cache/model-00001-of-000001.safetensors \
        --mm_vision_select_feature cls_patch \
        --mm_projector linear \
        --tune_vision_tower False \
        --tune_mm_projector True \
        --tune_language_model True \
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
        --learning_rate 5e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --report_to wandb 
        
# #!/bin/bash
# # ulimit -u 8192

# DEFAULT_RUN_NAME="ocr-qwen2-vl-7b-pretrain"
# DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=32
# DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

# STAGE_PATH=${1:-"runs/train/ocr-qwen2-vl-8b-align/model"}
# DATA_MIXTURE=${2:-"nvila-pretrain"}
# OUTPUT_DIR=${3:-"runs/train/ocr-qwen2-vl-8b-pretrain"}

# source scripts/setups/train.sh

# # NCCL_P2P_LEVEL=2
# # NCCL_P2P_DISABLE=1
# # NCCL_IB_TIMEOUT=22
# # TORCH_NCCL_BLOCKING_WAIT=0
# # NCCL_TIMEOUT=1800   # Increase timeout to 30 minutes
# # GPUS_PER_NODE=1

# torchrun \
#     --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
#     llava/train/train_mem.py \
#         --deepspeed scripts/zero3_new.json \
#         --model_name_or_path $STAGE_PATH \
#         --data_mixture $DATA_MIXTURE \
#         --vision_tower /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/checkpoints/sam_clip_ckpt/model_cache/model-00001-of-000001.safetensors \
#         --mm_vision_select_feature cls_patch \
#         --mm_projector linear \
#         --tune_vision_tower False \
#         --tune_mm_projector True \
#         --tune_language_model True \
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
#         --learning_rate 5e-5 \
#         --weight_decay 0. \
#         --warmup_ratio 0.03 \
#         --lr_scheduler_type cosine \
#         --logging_steps 1 \
#         --model_max_length 4096 \
#         --gradient_checkpointing True \
#         --dataloader_num_workers 4 \
#         --report_to wandb 
 



# #!/bin/bash
# # ulimit -u 8192
# DEFAULT_RUN_NAME="ocr-qwen2-vl-7b-pretrain-lora"
# DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=64
# DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

# STAGE_PATH=${1:-"runs/train/ocr-qwen2-vl-8b-align/model"}
# DATA_MIXTURE=${2:-"olmOCR-mix-pretrain"}
# OUTPUT_DIR=${3:-"runs/train/ocr-qwen2-vl-8b-pretrain"}

# source scripts/setups/train.sh
# export WANDB_MODE=offline
# export WANDB_SERVICE_WAIT=1200
# export WANDB_INIT_TIMEOUT=1200

# GPUS_PER_NODE=1
# CUDA_VISIBLE_DEVICES=0

# torchrun \
#     --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
#     llava/train/train_mem.py \
#         --deepspeed scripts/zero3_new.json \
#         --model_name_or_path $STAGE_PATH \
#         --data_mixture $DATA_MIXTURE \
#         --vision_tower /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/checkpoints/sam_clip_ckpt/model_cache/model-00001-of-000001.safetensors \
#         --mm_vision_select_feature cls_patch \
#         --mm_projector linear \
#         --tune_vision_tower False \
#         --tune_mm_projector True \
#         --tune_language_model True \
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
#         --learning_rate 5e-5 \
#         --weight_decay 0. \
#         --warmup_ratio 0.03 \
#         --lr_scheduler_type cosine \
#         --logging_steps 1 \
#         --model_max_length 4096 \
#         --gradient_checkpointing True \
#         --dataloader_num_workers 4 \
#         --report_to wandb \
#         --lora_enable True \
#         --lora_r 128 \
#         --lora_alpha 256 \
#         --lora_dropout 0.05 \
#         --lora_bias none \
#         --lora_llm True \
#         --lora_vt False
 


