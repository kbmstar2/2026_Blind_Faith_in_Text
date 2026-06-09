#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/iiixr/Documents/users/Byungmin/WordsOrVision/blind-faith-in-text/raw_code
cd "${ROOT}"
source /home/iiixr/anaconda3/etc/profile.d/conda.sh
conda activate wordsorvision

export CUDA_VISIBLE_DEVICES=GPU-908881af-34c5-90af-2108-e1720717127b
export TOKENIZERS_PARALLELISM=false

PY=/home/iiixr/anaconda3/envs/wordsorvision/bin/python
MODEL=llava-hf/llava-v1.6-vicuna-7b-hf
TAG=llava_next_7b_textvqa_conflict_dpo_train1500
DATA_ROOT=data/textvqa_type_matched_corruption_pilot_seed0_2000
OUT=results/textvqa_llava_conflict_dpo_train1500
DATASET=data/dpo_TextVQA_llava_next_7b_type_matched_conflict_seed0_train1500
REF_CACHE=${DATASET}_refcache
CKPT=checkpoints/llava_next_7b_textvqa_conflict_dpo_train1500_r16_beta01_lr1e5_300_ada
MERGED=${CKPT}_merged_tmp

mkdir -p logs/textvqa_llava_conflict_dpo "${OUT}"

log_step() {
  echo "[$(date)] [${TAG}] $*"
}

if [ ! -d "${DATA_ROOT}/corrupted" ] || [ ! -d "${DATA_ROOT}/match" ]; then
  log_step "missing ${DATA_ROOT}; build 2000-sample type-matched TextVQA conflict dataset"
  "${PY}" our_codes/make_textvqa_corrupted_pilot.py \
    --out_root "${DATA_ROOT}" \
    --max_samples 2000 \
    --seed 0 \
    --corruption_mode question_type_matched_answer
fi

if [ ! -d "${DATASET}" ]; then
  log_step "missing ${DATASET}; run mining script first"
  exit 1
fi

log_step "base heldout eval corrupted offset1500 n500"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/corrupted" \
  --model_type adapter \
  --model_name "${MODEL}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 500 \
  --sample_offset 1500 \
  --seed 0 \
  --out_file "${OUT}/${TAG}_base_corrupted_heldout500.jsonl"

log_step "base heldout eval match offset1500 n500"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/match" \
  --model_type adapter \
  --model_name "${MODEL}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 500 \
  --sample_offset 1500 \
  --seed 0 \
  --out_file "${OUT}/${TAG}_base_match_heldout500.jsonl"

log_step "train LLaVA-7B TextVQA conflict DPO with train1500-mined pairs"
rm -rf "${CKPT}" "${REF_CACHE}"
"${PY}" our_codes/train_dpo_llava_lora_precompute.py \
  --dataset_dir "${DATASET}" \
  --ref_cache_dir "${REF_CACHE}" \
  --model_name "${MODEL}" \
  --output_dir "${CKPT}" \
  --max_steps 300 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 1e-5 \
  --beta 0.1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --save_every 150

log_step "merge adapter"
rm -rf "${MERGED}"
"${PY}" our_codes/merge_llava_lora.py \
  --base_model "${MODEL}" \
  --adapter_dir "${CKPT}" \
  --output_dir "${MERGED}"

log_step "DPO heldout eval corrupted offset1500 n500"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/corrupted" \
  --model_type adapter \
  --model_name "${MERGED}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 500 \
  --sample_offset 1500 \
  --seed 0 \
  --out_file "${OUT}/${TAG}_dpo_corrupted_heldout500.jsonl"

log_step "DPO heldout eval match offset1500 n500"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/match" \
  --model_type adapter \
  --model_name "${MERGED}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 500 \
  --sample_offset 1500 \
  --seed 0 \
  --out_file "${OUT}/${TAG}_dpo_match_heldout500.jsonl"

log_step "analysis"
"${PY}" our_codes/measure_text_following_rate.py \
  "${OUT}/${TAG}_base_corrupted_heldout500.jsonl" \
  "${OUT}/${TAG}_dpo_corrupted_heldout500.jsonl" \
  > "${OUT}/${TAG}_dpo_corrupted_heldout500.text_following.txt"
"${PY}" our_codes/compare_eval_outputs.py \
  --base "${OUT}/${TAG}_base_corrupted_heldout500.jsonl" \
  --new "${OUT}/${TAG}_dpo_corrupted_heldout500.jsonl" \
  > "${OUT}/${TAG}_dpo_corrupted_heldout500_vs_base.txt"
"${PY}" our_codes/compare_eval_outputs.py \
  --base "${OUT}/${TAG}_base_match_heldout500.jsonl" \
  --new "${OUT}/${TAG}_dpo_match_heldout500.jsonl" \
  > "${OUT}/${TAG}_dpo_match_heldout500_vs_base.txt"

log_step "summaries"
cat "${OUT}/${TAG}_base_corrupted_heldout500.jsonl.summary.json"
cat "${OUT}/${TAG}_dpo_corrupted_heldout500.jsonl.summary.json"
cat "${OUT}/${TAG}_base_match_heldout500.jsonl.summary.json"
cat "${OUT}/${TAG}_dpo_match_heldout500.jsonl.summary.json"
cat "${OUT}/${TAG}_dpo_corrupted_heldout500.text_following.txt"
cat "${OUT}/${TAG}_dpo_corrupted_heldout500_vs_base.txt"
cat "${OUT}/${TAG}_dpo_match_heldout500_vs_base.txt"

case "${MERGED}" in
  checkpoints/*_merged_tmp) rm -rf "${MERGED}" ;;
  *) echo "refusing to delete ${MERGED}"; exit 1 ;;
esac

df -h .
log_step "done"
