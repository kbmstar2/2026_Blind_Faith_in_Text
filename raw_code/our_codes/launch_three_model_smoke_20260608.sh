#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/iiixr/Documents/users/Byungmin/WordsOrVision/blind-faith-in-text/raw_code
cd "${ROOT}"
source /home/iiixr/anaconda3/etc/profile.d/conda.sh
conda activate wordsorvision

mkdir -p logs/three_model_transfer results/three_model_transfer

run_smoke() {
  local gpu_uuid="$1"
  local tag="$2"
  local model="$3"
  local out="results/three_model_transfer/${tag}_base_corrupted_smoke20.jsonl"
  local log="logs/three_model_transfer/${tag}_smoke20.log"

  (
    set -euo pipefail
    cd "${ROOT}"
    source /home/iiixr/anaconda3/etc/profile.d/conda.sh
    conda activate wordsorvision
    export CUDA_VISIBLE_DEVICES="${gpu_uuid}"
    export TOKENIZERS_PARALLELISM=false
    echo "===== START ${tag} smoke at $(date) on ${gpu_uuid} ====="
    python hf_evaluator.py \
      --ds_name dal-289/word_or_vision \
      --subset DocVQA \
      --model_type adapter \
      --model_name "${model}" \
      --input_type text+image \
      --text_type corrupted \
      --max_samples 20 \
      --sample_offset 800 \
      --seed 0 \
      --out_file "${out}"
    cat "${out}.summary.json"
    echo "===== DONE ${tag} smoke at $(date) ====="
  ) > "${log}" 2>&1 &
  echo $! > "logs/three_model_transfer/${tag}_smoke20.pid"
}

run_smoke GPU-ebac977b-5d7f-afe6-e8ab-cc7ab6430066 qwen2vl7b Qwen/Qwen2-VL-7B-Instruct
run_smoke GPU-908881af-34c5-90af-2108-e1720717127b llava_next_7b llava-hf/llava-v1.6-vicuna-7b-hf
run_smoke GPU-82fa94af-7f79-c5aa-6d1b-c910ff15877e llava_next_13b llava-hf/llava-v1.6-vicuna-13b-hf

echo "Launched smoke tests:"
cat logs/three_model_transfer/*_smoke20.pid
