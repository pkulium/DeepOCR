# export HF_HOME=/lustre/hdd/LAS/wzhang-lab/mingl/cache
# export HUGGINGFACE_HUB_CACHE=/lustre/hdd/LAS/wzhang-lab/mingl/cache
# export TRANSFORMERS_CACHE=/lustre/hdd/LAS/wzhang-lab/mingl/cache
# export HF_DATASETS_CACHE=/lustre/hdd/LAS/wzhang-lab/mingl/cache

# MODEL_NAME=NVILA-8B-hf
# MODEL_ID=Efficient-Large-Model/$MODEL_NAME

# vila-eval \
#     --model-name $MODEL_NAME \
#     --model-path /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/VILA/runs/train/ocr-qwen2-vl-8b-pretrain-sam_clip/model \
#     --conv-mode auto \
#     --tags-include local \
#     --nproc-per-node 1


python llava/eval/omini_doc_bench.py \
  --model-path /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/VILA/runs/train/ocr-qwen2-vl-8b-pretrain-sam_clip_data_mix_without_syndog/model \
  --input-folder /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/data/OmniDocBench/images \
  --output-folder /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/VILA/runs/eval/ocr-qwen2-vl-8b-pretrain-sam_clip_data_mix_without_syndog/omini_doc_bench \
  --text "Free OCR." 


python llava/eval/olm_bench.py \
  --model-path /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/VILA/runs/train/ocr-qwen2-vl-8b-pretrain-sam_clip_data_mix_without_syndog/model \
  --input-folder /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/data/olmOCR-bench/bench_data/pngs \
  --output-folder /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/VILA/runs/eval/ocr-qwen2-vl-8b-pretrain-sam_clip_data_mix_without_syndog/olm_bench \
  --text "Free OCR." 

# python llava/eval/olm_bench.py \
#   --model-path /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/VILA/runs/train/ocr-qwen2-vl-8b-pretrain-sam_clip/model \
#   --input-folder /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/data/Fox_benchmark_data/focus_benchmark_test \
#   --output-folder /lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/VILA/runs/eval/NVILA-8B-hf/fox_bench \
#   --text "Free OCR." 

 