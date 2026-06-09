#!/usr/bin/env bash
set -euo pipefail

cd /home/iiixr/Documents/users/Byungmin/WordsOrVision/blind-faith-in-text/raw_code
source /home/iiixr/anaconda3/etc/profile.d/conda.sh
conda activate wordsorvision

export CUDA_VISIBLE_DEVICES=GPU-ebac977b-5d7f-afe6-e8ab-cc7ab6430066
export TOKENIZERS_PARALLELISM=false

PY=/home/iiixr/anaconda3/envs/wordsorvision/bin/python
BASE=Qwen/Qwen3-VL-8B-Instruct
ADAPTER=checkpoints/qwen3vl8b_dpo_span_split_train800_r16_beta01_lr1e5_700_ada
MERGED=${ADAPTER}_merged
DATA_ROOT=data/textvqa_corruption_pilot_seed0_200
OUT=results/textvqa_pilot

mkdir -p logs/textvqa_pilot "${OUT}"

log_step() {
  echo "[$(date)] $*"
}

log_step "ensure TextVQA pilot dataset exists"
if [ ! -d "${DATA_ROOT}/corrupted" ] || [ ! -d "${DATA_ROOT}/match" ]; then
  "${PY}" our_codes/make_textvqa_corrupted_pilot.py \
    --out_root "${DATA_ROOT}" --max_samples 200 --seed 0
fi

log_step "baseline Qwen3 TextVQA corrupted200"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/corrupted" \
  --model_type adapter \
  --model_name "${BASE}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 200 \
  --sample_offset 0 \
  --seed 0 \
  --out_file "${OUT}/qwen3vl8b_base_textvqa_corrupted200.jsonl"

log_step "baseline Qwen3 TextVQA match200"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/match" \
  --model_type adapter \
  --model_name "${BASE}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 200 \
  --sample_offset 0 \
  --seed 0 \
  --out_file "${OUT}/qwen3vl8b_base_textvqa_match200.jsonl"

log_step "merge DPO r16 beta0.1"
rm -rf "${MERGED}"
"${PY}" our_codes/merge_qwen2vl_lora.py \
  --base_model "${BASE}" \
  --adapter_dir "${ADAPTER}" \
  --output_dir "${MERGED}"

log_step "DPO Qwen3 TextVQA corrupted200"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/corrupted" \
  --model_type adapter \
  --model_name "${MERGED}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 200 \
  --sample_offset 0 \
  --seed 0 \
  --out_file "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_corrupted200.jsonl"

log_step "DPO Qwen3 TextVQA match200"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/match" \
  --model_type adapter \
  --model_name "${MERGED}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 200 \
  --sample_offset 0 \
  --seed 0 \
  --out_file "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_match200.jsonl"

log_step "analysis"
"${PY}" our_codes/measure_text_following_rate.py \
  "${OUT}/qwen3vl8b_base_textvqa_corrupted200.jsonl" \
  "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_corrupted200.jsonl" \
  > "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_corrupted200.text_following.txt"
"${PY}" our_codes/compare_eval_outputs.py \
  --base "${OUT}/qwen3vl8b_base_textvqa_corrupted200.jsonl" \
  --new "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_corrupted200.jsonl" \
  > "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_corrupted200_vs_base.txt"
"${PY}" our_codes/compare_eval_outputs.py \
  --base "${OUT}/qwen3vl8b_base_textvqa_match200.jsonl" \
  --new "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_match200.jsonl" \
  > "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_match200_vs_base.txt"

log_step "summaries"
cat "${OUT}/qwen3vl8b_base_textvqa_corrupted200.jsonl.summary.json"
cat "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_corrupted200.jsonl.summary.json"
cat "${OUT}/qwen3vl8b_base_textvqa_match200.jsonl.summary.json"
cat "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_match200.jsonl.summary.json"
cat "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_corrupted200.text_following.txt"
cat "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_corrupted200_vs_base.txt"
cat "${OUT}/qwen3vl8b_dpo_docvqa_span_r16_textvqa_match200_vs_base.txt"

case "${MERGED}" in
  checkpoints/*_merged) rm -rf "${MERGED}" ;;
  *) echo "refusing to delete ${MERGED}"; exit 1 ;;
esac

log_step "done"
df -h .
