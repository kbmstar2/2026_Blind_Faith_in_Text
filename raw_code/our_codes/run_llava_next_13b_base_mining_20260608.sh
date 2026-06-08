#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/iiixr/Documents/users/Byungmin/WordsOrVision/blind-faith-in-text/raw_code
cd "${ROOT}"
source /home/iiixr/anaconda3/etc/profile.d/conda.sh
conda activate wordsorvision

export CUDA_VISIBLE_DEVICES=GPU-82fa94af-7f79-c5aa-6d1b-c910ff15877e
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/three_model_transfer results/three_model_transfer

model=llava-hf/llava-v1.6-vicuna-13b-hf
tag=llava_next_13b
dataset=data/dpo_DocVQA_${tag}_text_follow_span_seed0_train800

echo "===== START ${tag} base/mining pipeline at $(date) ====="
df -h .

python hf_evaluator.py \
  --ds_name dal-289/word_or_vision \
  --subset DocVQA \
  --model_type adapter \
  --model_name "${model}" \
  --input_type text+image \
  --text_type corrupted \
  --max_samples 800 \
  --sample_offset 0 \
  --seed 0 \
  --out_file results/three_model_transfer/${tag}_base_corrupted_seed0_train800.jsonl

python hf_evaluator.py \
  --ds_name dal-289/word_or_vision \
  --subset DocVQA \
  --model_type adapter \
  --model_name "${model}" \
  --input_type text+image \
  --text_type corrupted \
  --max_samples 200 \
  --sample_offset 800 \
  --seed 0 \
  --out_file results/three_model_transfer/${tag}_base_corrupted_heldout200.jsonl

python our_codes/make_text_following_span_dpo.py \
  --eval_file results/three_model_transfer/${tag}_base_corrupted_seed0_train800.jsonl \
  --out_dir "${dataset}" \
  --subset DocVQA \
  --text_type corrupted \
  --seed 0

python our_codes/measure_text_following_rate.py \
  results/three_model_transfer/${tag}_base_corrupted_heldout200.jsonl \
  > results/three_model_transfer/${tag}_base_corrupted_heldout200.text_following.txt || true

echo "----- summaries -----"
cat results/three_model_transfer/${tag}_base_corrupted_seed0_train800.jsonl.summary.json
cat results/three_model_transfer/${tag}_base_corrupted_heldout200.jsonl.summary.json
cat results/three_model_transfer/${tag}_base_corrupted_heldout200.text_following.txt || true
echo "dataset rows:"
python - <<'PY'
from datasets import load_from_disk
ds = load_from_disk("data/dpo_DocVQA_llava_next_13b_text_follow_span_seed0_train800")
print(ds)
PY

df -h .
echo "===== DONE ${tag} base/mining pipeline at $(date) ====="
