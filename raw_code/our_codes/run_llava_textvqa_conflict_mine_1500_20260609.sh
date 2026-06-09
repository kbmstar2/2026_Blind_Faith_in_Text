#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/iiixr/Documents/users/Byungmin/WordsOrVision/blind-faith-in-text/raw_code
cd "${ROOT}"
source /home/iiixr/anaconda3/etc/profile.d/conda.sh
conda activate wordsorvision

export CUDA_VISIBLE_DEVICES=GPU-82fa94af-7f79-c5aa-6d1b-c910ff15877e
export TOKENIZERS_PARALLELISM=false

PY=/home/iiixr/anaconda3/envs/wordsorvision/bin/python
MODEL=llava-hf/llava-v1.6-vicuna-7b-hf
TAG=llava_next_7b_textvqa_conflict_mine1500
DATA_ROOT=data/textvqa_type_matched_corruption_pilot_seed0_2000
OUT=results/textvqa_llava_conflict_dpo_mine1500
DATASET=data/dpo_TextVQA_llava_next_7b_type_matched_conflict_seed0_train1500

mkdir -p logs/textvqa_llava_conflict_dpo "${OUT}"

log_step() {
  echo "[$(date)] [${TAG}] $*"
}

log_step "build 2000-sample type-matched TextVQA conflict dataset"
if [ ! -d "${DATA_ROOT}/corrupted" ] || [ ! -d "${DATA_ROOT}/match" ]; then
  "${PY}" our_codes/make_textvqa_corrupted_pilot.py \
    --out_root "${DATA_ROOT}" \
    --max_samples 2000 \
    --seed 0 \
    --corruption_mode question_type_matched_answer
fi

log_step "base train-mining eval corrupted first1500"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/corrupted" \
  --model_type adapter \
  --model_name "${MODEL}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 1500 \
  --sample_offset 0 \
  --seed 0 \
  --out_file "${OUT}/${TAG}_base_corrupted_train1500.jsonl"

log_step "make LLaVA-specific TextVQA DPO dataset from mined errors"
rm -rf "${DATASET}"
"${PY}" our_codes/make_local_textvqa_dpo_from_eval_errors.py \
  --eval_file "${OUT}/${TAG}_base_corrupted_train1500.jsonl" \
  --dataset_dir "${DATA_ROOT}/corrupted" \
  --out_dir "${DATASET}" \
  --test_size 0.1 \
  --seed 0

log_step "summary"
cat "${OUT}/${TAG}_base_corrupted_train1500.jsonl.summary.json"
df -h .
log_step "done"
