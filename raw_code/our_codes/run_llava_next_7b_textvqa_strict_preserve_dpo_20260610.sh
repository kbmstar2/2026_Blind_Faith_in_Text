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
TAG=llava_next_7b_textvqa_strict_preserve_dpo
DATA_ROOT=data/textvqa_type_matched_corruption_pilot_seed0_2000
OUT=results/textvqa_llava_strict_preserve_dpo
DATASET=data/dpo_TextVQA_llava_next_7b_strict_preserve_seed0_train1500
REF_CACHE=${DATASET}_refcache
CKPT=checkpoints/llava_next_7b_textvqa_strict_preserve_dpo_r16_beta01_lr1e5_300_ada
MERGED=${CKPT}_merged_tmp

mkdir -p logs/textvqa_llava_conflict_dpo "${OUT}"

log_step() {
  echo "[$(date)] [${TAG}] $*"
}

if [ ! -d "${DATASET}" ]; then
  log_step "build strict+preserve DPO dataset"
  "${PY}" our_codes/make_local_textvqa_dpo_strict_preserve.py \
    --eval_file results/textvqa_llava_conflict_dpo_mine1500/llava_next_7b_textvqa_conflict_mine1500_base_corrupted_train1500.jsonl \
    --dataset_dir "${DATA_ROOT}/corrupted" \
    --out_dir "${DATASET}" \
    --test_size 0.1 \
    --seed 0 \
    --preserve_weight 0.5 \
    --max_preserve_per_correction 1.0
fi

log_step "train LLaVA-7B strict+preserve TextVQA DPO"
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
  results/textvqa_llava_conflict_dpo_train1500/llava_next_7b_textvqa_conflict_dpo_train1500_base_corrupted_heldout500.jsonl \
  "${OUT}/${TAG}_dpo_corrupted_heldout500.jsonl" \
  > "${OUT}/${TAG}_dpo_corrupted_heldout500.text_following.txt"
"${PY}" our_codes/compare_eval_outputs.py \
  --base results/textvqa_llava_conflict_dpo_train1500/llava_next_7b_textvqa_conflict_dpo_train1500_base_corrupted_heldout500.jsonl \
  --new "${OUT}/${TAG}_dpo_corrupted_heldout500.jsonl" \
  > "${OUT}/${TAG}_dpo_corrupted_heldout500_vs_base.txt"
"${PY}" our_codes/compare_eval_outputs.py \
  --base results/textvqa_llava_conflict_dpo_train1500/llava_next_7b_textvqa_conflict_dpo_train1500_base_match_heldout500.jsonl \
  --new "${OUT}/${TAG}_dpo_match_heldout500.jsonl" \
  > "${OUT}/${TAG}_dpo_match_heldout500_vs_base.txt"

log_step "summaries"
cat results/textvqa_llava_conflict_dpo_train1500/llava_next_7b_textvqa_conflict_dpo_train1500_base_corrupted_heldout500.jsonl.summary.json
cat "${OUT}/${TAG}_dpo_corrupted_heldout500.jsonl.summary.json"
cat results/textvqa_llava_conflict_dpo_train1500/llava_next_7b_textvqa_conflict_dpo_train1500_base_match_heldout500.jsonl.summary.json
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
