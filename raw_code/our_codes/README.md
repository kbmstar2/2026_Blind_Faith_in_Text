# Our Codes

This folder contains the code and notes added for the text-bias reduction experiments.
The original upstream code is kept in `raw_code/` as much as possible.

## DPO Track

- `TEXT_FOLLOWING_SPAN_DPO.md`: summary of the DPO experiment and main results.
- `make_text_following_span_dpo.py`: builds high-precision text-following-error DPO data from evaluation outputs.
- `train_dpo_qwen2vl_lora.py`: Qwen2/Qwen3-VL LoRA DPO trainer.
- `merge_qwen2vl_lora.py`: merges a trained LoRA adapter into the base model.
- `compare_eval_outputs.py`: compares corrected and regressed predictions between two evaluation files.
- `measure_text_following_rate.py`: measures how often predictions are copied from corrupted auxiliary text.

## GRPO Track

- `grpo/train_grpo_qwen3vl_lora.py`: GRPO trainer for Qwen2/Qwen3-VL.
- `grpo/GRPO_EXPERIMENTS.md`: GRPO pilot results and next experiment plan.

DPO and GRPO are intentionally kept separate so their effects can be compared directly. Chained DPO -> GRPO runs should be labeled separately from pure DPO or pure GRPO.

## Upstream Compatibility Note

`raw_code/hf_evaluator.py` has one small compatibility change: `--sample_offset`, used to evaluate held-out shuffled slices without reusing the train-mining subset.
