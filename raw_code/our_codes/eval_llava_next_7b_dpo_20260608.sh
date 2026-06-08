#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/iiixr/Documents/users/Byungmin/WordsOrVision/blind-faith-in-text/raw_code
cd "${ROOT}"
source /home/iiixr/anaconda3/etc/profile.d/conda.sh
conda activate wordsorvision

export CUDA_VISIBLE_DEVICES=GPU-908881af-34c5-90af-2108-e1720717127b
export TOKENIZERS_PARALLELISM=false

tag=llava_next_7b
merged=checkpoints/${tag}_dpo_span_seed0_train800_r16_beta01_lr1e5_300_ada_merged

echo "===== START ${tag} DPO eval resume at $(date) ====="

python hf_evaluator.py \
  --ds_name dal-289/word_or_vision \
  --subset DocVQA \
  --model_type adapter \
  --model_name "${merged}" \
  --input_type text+image \
  --text_type corrupted \
  --max_samples 200 \
  --sample_offset 800 \
  --seed 0 \
  --out_file results/three_model_transfer/${tag}_dpo_span_r16_300_corrupted_heldout200.jsonl

python hf_evaluator.py \
  --ds_name dal-289/word_or_vision \
  --subset DocVQA \
  --model_type adapter \
  --model_name "${merged}" \
  --input_type text+image \
  --text_type match \
  --max_samples 100 \
  --sample_offset 800 \
  --seed 0 \
  --out_file results/three_model_transfer/${tag}_dpo_span_r16_300_match_heldout100.jsonl

python hf_evaluator.py \
  --ds_name dal-289/word_or_vision \
  --subset DocVQA \
  --model_type adapter \
  --model_name "${merged}" \
  --input_type text+image \
  --text_type irrelevant \
  --max_samples 100 \
  --sample_offset 800 \
  --seed 0 \
  --out_file results/three_model_transfer/${tag}_dpo_span_r16_300_irrelevant_heldout100.jsonl

python our_codes/compare_eval_outputs.py \
  --base results/three_model_transfer/${tag}_base_corrupted_heldout200.jsonl \
  --new results/three_model_transfer/${tag}_dpo_span_r16_300_corrupted_heldout200.jsonl \
  > results/three_model_transfer/${tag}_dpo_span_r16_300_vs_base_heldout200.txt || true

python our_codes/measure_text_following_rate.py \
  results/three_model_transfer/${tag}_base_corrupted_heldout200.jsonl \
  results/three_model_transfer/${tag}_dpo_span_r16_300_corrupted_heldout200.jsonl \
  > results/three_model_transfer/${tag}_dpo_span_r16_300_corrupted_heldout200.text_following.txt || true

echo "----- summaries -----"
cat results/three_model_transfer/${tag}_base_corrupted_heldout200.jsonl.summary.json
cat results/three_model_transfer/${tag}_dpo_span_r16_300_corrupted_heldout200.jsonl.summary.json
cat results/three_model_transfer/${tag}_dpo_span_r16_300_match_heldout100.jsonl.summary.json
cat results/three_model_transfer/${tag}_dpo_span_r16_300_irrelevant_heldout100.jsonl.summary.json
cat results/three_model_transfer/${tag}_dpo_span_r16_300_vs_base_heldout200.txt || true
cat results/three_model_transfer/${tag}_dpo_span_r16_300_corrupted_heldout200.text_following.txt || true

case "${merged}" in
  checkpoints/*_merged) rm -rf "${merged}" ;;
  *) echo "refusing to delete ${merged}"; exit 1 ;;
esac

df -h .
echo "===== DONE ${tag} DPO eval resume at $(date) ====="
