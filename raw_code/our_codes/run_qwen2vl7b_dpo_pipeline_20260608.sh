#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/iiixr/Documents/users/Byungmin/WordsOrVision/blind-faith-in-text/raw_code
cd "${ROOT}"
source /home/iiixr/anaconda3/etc/profile.d/conda.sh
conda activate wordsorvision

export CUDA_VISIBLE_DEVICES=GPU-ebac977b-5d7f-afe6-e8ab-cc7ab6430066
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/three_model_transfer results/three_model_transfer

model=Qwen/Qwen2-VL-7B-Instruct
tag=qwen2vl7b
dataset=data/dpo_DocVQA_${tag}_text_follow_span_seed0_train800
ckpt=checkpoints/${tag}_dpo_span_seed0_train800_r16_beta01_lr1e5_700_ada
merged=${ckpt}_merged

echo "===== START ${tag} DPO pipeline at $(date) ====="
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

python our_codes/make_text_following_span_dpo.py \
  --eval_file results/three_model_transfer/${tag}_base_corrupted_seed0_train800.jsonl \
  --out_dir "${dataset}" \
  --subset DocVQA \
  --text_type corrupted \
  --seed 0

python our_codes/train_dpo_qwen2vl_lora.py \
  --dataset_dir "${dataset}" \
  --model_name "${model}" \
  --output_dir "${ckpt}" \
  --max_steps 700 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 1e-5 \
  --beta 0.1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_pixels 401408 \
  --save_every 350

rm -rf "${merged}"
python our_codes/merge_qwen2vl_lora.py \
  --base_model "${model}" \
  --adapter_dir "${ckpt}" \
  --output_dir "${merged}"

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
  --out_file results/three_model_transfer/${tag}_dpo_span_r16_corrupted_heldout200.jsonl

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
  --out_file results/three_model_transfer/${tag}_dpo_span_r16_match_heldout100.jsonl

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
  --out_file results/three_model_transfer/${tag}_dpo_span_r16_irrelevant_heldout100.jsonl

python our_codes/compare_eval_outputs.py \
  --base results/three_model_transfer/${tag}_base_corrupted_heldout200.jsonl \
  --new results/three_model_transfer/${tag}_dpo_span_r16_corrupted_heldout200.jsonl \
  > results/three_model_transfer/${tag}_dpo_span_r16_vs_base_heldout200.txt || true

python our_codes/measure_text_following_rate.py \
  results/three_model_transfer/${tag}_base_corrupted_heldout200.jsonl \
  results/three_model_transfer/${tag}_dpo_span_r16_corrupted_heldout200.jsonl \
  > results/three_model_transfer/${tag}_dpo_span_r16_corrupted_heldout200.text_following.txt || true

echo "----- summaries -----"
cat results/three_model_transfer/${tag}_base_corrupted_seed0_train800.jsonl.summary.json
cat results/three_model_transfer/${tag}_base_corrupted_heldout200.jsonl.summary.json
cat results/three_model_transfer/${tag}_dpo_span_r16_corrupted_heldout200.jsonl.summary.json
cat results/three_model_transfer/${tag}_dpo_span_r16_match_heldout100.jsonl.summary.json
cat results/three_model_transfer/${tag}_dpo_span_r16_irrelevant_heldout100.jsonl.summary.json
cat results/three_model_transfer/${tag}_dpo_span_r16_corrupted_heldout200.text_following.txt || true

case "${merged}" in
  checkpoints/*_merged) rm -rf "${merged}" ;;
  *) echo "refusing to delete ${merged}"; exit 1 ;;
esac

df -h .
echo "===== DONE ${tag} DPO pipeline at $(date) ====="
