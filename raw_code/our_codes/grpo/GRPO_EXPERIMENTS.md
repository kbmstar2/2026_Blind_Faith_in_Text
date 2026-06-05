# GRPO Experiments

This note keeps the GRPO line separate from the DPO experiments so the two methods can be compared cleanly.

## Purpose

DPO and GRPO should be treated as separate interventions:

- DPO: supervised preference learning from mined text-following error pairs.
- GRPO: online sampled-answer optimization with an explicit reward function.

The first GRPO trials were initialized from the DPO checkpoint. That is useful as a pilot, but it should be reported as DPO + GRPO rather than pure GRPO. For a cleaner comparison, run both of these settings:

- DPO only: base model -> DPO.
- GRPO only: base model -> GRPO, using the same train/test split.
- Optional chained setting: base model -> DPO -> GRPO.

## Existing Pilot Results

Model: `Qwen/Qwen3-VL-8B-Instruct`

Training setup:

- Starting point: DPO checkpoint.
- Dataset: DocVQA text-following/span training data.
- GRPO length: 100 pilot steps.
- LoRA rank 8, alpha 16, lr `2e-6`.

DocVQA corrupted 1000:

| Variant | Accuracy | Incorrect aux-hit | Preservation |
|---|---:|---:|---|
| Base Qwen3-VL-8B | 0.724 | 90.2% | - |
| DPO only, base-error pairs | 0.790 | 84.3% | match 1.00, irrelevant 0.96 |
| DPO + GRPO v1 pilot | 0.789 | 83.4% | match 1.00, irrelevant 0.96 |
| DPO + GRPO v2 pilot | 0.786 | 83.6% | match 1.00, irrelevant 0.96 |
| DPO + GRPO v3 pilot | 0.789 | 83.9% | match 1.00, irrelevant 0.96 |
| Text-following span DPO | 0.895 | 61.0% | match 1.00, irrelevant 0.96 |

Interpretation: the early DPO + GRPO pilots did not improve over the DPO baseline enough to justify mixing them into the DPO claim. GRPO is still worth testing separately, but the reward must be more carefully aligned with text-bias reduction.

## Current Reward Sketch

`train_grpo_qwen3vl_lora.py` gives reward for:

- matching the image-grounded chosen answer;
- avoiding the rejected answer mined from corrupted auxiliary text;
- avoiding high-confidence misleading spans from the corrupted auxiliary text;
- producing a short answer-like completion.

This is a starting point, not a final reward. The next version should separate reward terms in logs so we can see whether improvements come from correctness, lower text copying, or just shorter outputs.

## Recommended Next Experiments

1. GRPO-only on the disjoint split.

   Use the same train-mining and held-out split as text-following span DPO:

   - train-mining: seed 0, offset 0, 800 examples;
   - held-out test: seed 0, offset 800, 200 examples.

2. Span-focused reward.

   Penalize generated answers that match any mined misleading span from corrupted auxiliary text, not only the single base rejected answer.

3. Multi-objective reporting.

   Report all of these on the held-out test slice:

   - accuracy;
   - incorrect aux-hit;
   - total pred-in-aux;
   - corrected/regressed vs base;
   - match/irrelevant preservation.

## Example Command

```bash
CUDA_VISIBLE_DEVICES=0 python our_codes/grpo/train_grpo_qwen3vl_lora.py \
  --dataset_dir data/dpo_DocVQA_text_follow_span_seed0_train800 \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --output_dir checkpoints/qwen3vl8b_grpo_only_span_split_r8_lr2e6_100_ada \
  --max_steps 100 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 2e-6 \
  --beta_kl 0.02 \
  --num_generations 4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --max_pixels 401408
```
