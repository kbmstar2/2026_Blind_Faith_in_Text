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

## Disjoint Split Results

Protocol:

- train-mining: DocVQA corrupted, seed 0, offset 0, 800 examples;
- held-out test: DocVQA corrupted, seed 0, offset 800, 200 examples;
- preservation: DocVQA match 100 and irrelevant 100;
- GPU: RTX 6000 Ada.

Held-out corrupted 200:

| Variant | Init | Steps | LR | Accuracy | Incorrect aux-hit | Corrected / Regressed vs base |
|---|---|---:|---:|---:|---:|---:|
| Base Qwen3-VL-8B | - | - | - | 0.715 | 89.5% | - |
| GRPO-only v1 | base | 100 | 2e-6 | 0.710 | 89.7% | 0 / 1 |
| GRPO-only v2 | base | 300 | 1e-5 | 0.705 | 89.8% | 0 / 2 |
| GRPO-only v3 dense reward | base | 300 | 8e-6 | 0.705 | 89.8% | 0 / 2 |
| Text-following span DPO | base | 700 | 1e-5 | 0.900 | 55.0% | 37 / 0 |
| Span DPO -> GRPO v2 | span DPO | 150 | 2e-6 | 0.900 | 60.0% | 37 / 0 |

Preservation:

| Variant | Match 100 | Irrelevant 100 |
|---|---:|---:|
| GRPO-only v1 | 1.00 | 0.96 |
| GRPO-only v2 | 1.00 | 0.96 |
| GRPO-only v3 dense reward | 1.00 | 0.96 |
| Span DPO -> GRPO v2 | 1.00 | 0.96 |

Training-signal observation:

- GRPO-only v2 generated correct sampled answers only about 0-15% of the time during training.
- Its predictions barely moved from base: 196/200 held-out predictions were unchanged, with 0 corrections and 2 regressions.
- GRPO-only v3 used a denser reward (`num_generations=8`, `temperature=1.0`, stronger rejected/span/aux-copy penalties, lower KL) and did produce more diagnostic reward signal during training: `correct_gen` was often 1-12.5%, and rejected/aux-copy matches were frequently observed. However, held-out greedy behavior still barely moved: 196/200 predictions were unchanged, with 0 corrections and 2 regressions. Incorrect aux-hit stayed at 89.8%.
- DPO -> GRPO v2 generated more correct samples during training, but it did not improve held-out accuracy over DPO. It changed only 3/200 held-out predictions relative to DPO, with 0 corrections and 0 regressions.

Interpretation: making the sampled reward denser is not enough by itself. Pure GRPO still fails to overcome the base model's text-following behavior in greedy evaluation. DPO remains the effective intervention. For GRPO to become competitive, the next change should alter the optimization target more directly, for example by anchoring each prompt with explicit chosen/rejected candidate log-prob rewards or by using a verifier-style reward over constrained candidate answers rather than relying only on free-form sampled completions.

## Current Reward Sketch

`train_grpo_qwen3vl_lora.py` gives reward for:

- matching the image-grounded chosen answer;
- avoiding the rejected answer mined from corrupted auxiliary text;
- avoiding high-confidence misleading spans from the corrupted auxiliary text;
- avoiding generated answers that appear in the corrupted auxiliary text;
- producing a short answer-like completion.

The trainer now logs reward components (`correct_gen`, `rejected`, `span`, `aux`, `answer_like`) so failures can be diagnosed instead of only reading final accuracy.

## Recommended Next Experiments

1. Repeat the disjoint split with additional seeds.

   The seed 0 split is strong but still only one held-out partition. Repeat seed 1 and seed 2 before treating the DPO advantage as stable.

2. Move beyond free-form sampled-answer GRPO.

   The dense-reward v3 run showed that more sampled reward signal still did not change greedy held-out behavior. Candidate directions:

   - use candidate-constrained GRPO: sample/rank among chosen, rejected, and mined misleading spans instead of only free-form completions;
   - add a direct log-prob margin reward for chosen over rejected candidates inside the GRPO objective;
   - reward image-grounded answer equivalence using ANLS/token similarity, not only exact match;
   - add a curriculum: begin from easier/high-confidence rows where the base model samples the correct answer at least once.

3. Span-focused penalty.

   Penalize generated answers that match any mined misleading span from corrupted auxiliary text, not only the single base rejected answer.

4. Multi-objective reporting.

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


Dense GRPO-only v3 command:

```bash
CUDA_VISIBLE_DEVICES=0 python our_codes/grpo/train_grpo_qwen3vl_lora.py \
  --dataset_dir data/dpo_DocVQA_text_follow_span_seed0_train800 \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --output_dir checkpoints/qwen3vl8b_grpo_only_v3_dense_docvqa_split_r8_lr8e6_300_ada \
  --max_steps 300 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 8e-6 \
  --beta_kl 0.01 \
  --num_generations 8 \
  --temperature 1.0 \
  --top_p 0.95 \
  --correct_reward 3.0 \
  --wrong_penalty 0.05 \
  --rejected_penalty 2.0 \
  --span_penalty 1.2 \
  --aux_copy_penalty 1.2 \
  --aux_free_reward 0.25 \
  --empty_penalty 0.8 \
  --format_reward 0.1 \
  --long_penalty 0.2 \
  --lora_r 8 \
  --lora_alpha 16 \
  --max_pixels 401408
```
