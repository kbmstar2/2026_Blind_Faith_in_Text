#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 {qwen2vl7b|llava_next_7b|llava_next_13b}" >&2
  exit 2
fi

ROOT=/home/iiixr/Documents/users/Byungmin/WordsOrVision/blind-faith-in-text/raw_code
cd "${ROOT}"
source /home/iiixr/anaconda3/etc/profile.d/conda.sh
conda activate wordsorvision

export TOKENIZERS_PARALLELISM=false

PY=/home/iiixr/anaconda3/envs/wordsorvision/bin/python
DATA_ROOT=data/textvqa_type_matched_corruption_pilot_seed0_200
OUT=results/textvqa_type_matched_transfer
mkdir -p logs/textvqa_type_matched_transfer "${OUT}"

tag="$1"
case "${tag}" in
  qwen2vl7b)
    export CUDA_VISIBLE_DEVICES=GPU-ebac977b-5d7f-afe6-e8ab-cc7ab6430066
    base_model=Qwen/Qwen2-VL-7B-Instruct
    adapter=checkpoints/qwen2vl7b_dpo_span_seed0_train800_r16_beta01_lr1e5_700_ada
    merge_script=our_codes/merge_qwen2vl_lora.py
    ;;
  llava_next_7b)
    export CUDA_VISIBLE_DEVICES=GPU-908881af-34c5-90af-2108-e1720717127b
    base_model=llava-hf/llava-v1.6-vicuna-7b-hf
    adapter=checkpoints/llava_next_7b_dpo_span_seed0_train800_r16_beta01_lr1e5_300_ada
    merge_script=our_codes/merge_llava_lora.py
    ;;
  llava_next_13b)
    export CUDA_VISIBLE_DEVICES=GPU-82fa94af-7f79-c5aa-6d1b-c910ff15877e
    base_model=llava-hf/llava-v1.6-vicuna-13b-hf
    adapter=checkpoints/llava_next_13b_dpo_span_seed0_train800_r16_beta01_lr1e5_300_ada
    merge_script=our_codes/merge_llava_lora.py
    ;;
  *)
    echo "unknown tag: ${tag}" >&2
    exit 2
    ;;
esac

merged=${adapter}_merged_textvqa_tmp

log_step() {
  echo "[$(date)] [${tag}] $*"
}

log_step "ensure TextVQA question-type-matched dataset exists"
if [ ! -d "${DATA_ROOT}/corrupted" ] || [ ! -d "${DATA_ROOT}/match" ]; then
  "${PY}" our_codes/make_textvqa_corrupted_pilot.py \
    --out_root "${DATA_ROOT}" \
    --max_samples 200 \
    --seed 0 \
    --corruption_mode question_type_matched_answer
fi

log_step "base corrupted"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/corrupted" \
  --model_type adapter \
  --model_name "${base_model}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 200 \
  --sample_offset 0 \
  --seed 0 \
  --out_file "${OUT}/${tag}_base_textvqa_type_matched_corrupted200.jsonl"

log_step "base match"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/match" \
  --model_type adapter \
  --model_name "${base_model}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 200 \
  --sample_offset 0 \
  --seed 0 \
  --out_file "${OUT}/${tag}_base_textvqa_type_matched_match200.jsonl"

log_step "merge DPO adapter"
rm -rf "${merged}"
"${PY}" "${merge_script}" \
  --base_model "${base_model}" \
  --adapter_dir "${adapter}" \
  --output_dir "${merged}"

log_step "DPO corrupted"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/corrupted" \
  --model_type adapter \
  --model_name "${merged}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 200 \
  --sample_offset 0 \
  --seed 0 \
  --out_file "${OUT}/${tag}_dpo_textvqa_type_matched_corrupted200.jsonl"

log_step "DPO match"
"${PY}" hf_evaluator.py \
  --ds_name "${DATA_ROOT}/match" \
  --model_type adapter \
  --model_name "${merged}" \
  --input_type text+image \
  --text_type "" \
  --max_samples 200 \
  --sample_offset 0 \
  --seed 0 \
  --out_file "${OUT}/${tag}_dpo_textvqa_type_matched_match200.jsonl"

log_step "analysis"
"${PY}" our_codes/measure_text_following_rate.py \
  "${OUT}/${tag}_base_textvqa_type_matched_corrupted200.jsonl" \
  "${OUT}/${tag}_dpo_textvqa_type_matched_corrupted200.jsonl" \
  > "${OUT}/${tag}_dpo_textvqa_type_matched_corrupted200.text_following.txt"
"${PY}" our_codes/compare_eval_outputs.py \
  --base "${OUT}/${tag}_base_textvqa_type_matched_corrupted200.jsonl" \
  --new "${OUT}/${tag}_dpo_textvqa_type_matched_corrupted200.jsonl" \
  > "${OUT}/${tag}_dpo_textvqa_type_matched_corrupted200_vs_base.txt"
"${PY}" our_codes/compare_eval_outputs.py \
  --base "${OUT}/${tag}_base_textvqa_type_matched_match200.jsonl" \
  --new "${OUT}/${tag}_dpo_textvqa_type_matched_match200.jsonl" \
  > "${OUT}/${tag}_dpo_textvqa_type_matched_match200_vs_base.txt"

log_step "summaries"
cat "${OUT}/${tag}_base_textvqa_type_matched_corrupted200.jsonl.summary.json"
cat "${OUT}/${tag}_dpo_textvqa_type_matched_corrupted200.jsonl.summary.json"
cat "${OUT}/${tag}_base_textvqa_type_matched_match200.jsonl.summary.json"
cat "${OUT}/${tag}_dpo_textvqa_type_matched_match200.jsonl.summary.json"
cat "${OUT}/${tag}_dpo_textvqa_type_matched_corrupted200.text_following.txt"
cat "${OUT}/${tag}_dpo_textvqa_type_matched_corrupted200_vs_base.txt"
cat "${OUT}/${tag}_dpo_textvqa_type_matched_match200_vs_base.txt"

case "${merged}" in
  checkpoints/*_merged_textvqa_tmp) rm -rf "${merged}" ;;
  *) echo "refusing to delete ${merged}"; exit 1 ;;
esac

df -h .
log_step "done"
