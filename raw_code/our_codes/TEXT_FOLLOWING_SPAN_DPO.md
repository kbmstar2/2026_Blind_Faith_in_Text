# Text-Following Span DPO

This note summarizes a lightweight preference-tuning experiment for reducing text bias in Word-or-Vision style corrupted-context VQA.

## Idea

Not every incorrect answer is a text-bias error. We focus on high-precision text-following errors:

1. Evaluate the base VLM under corrupted auxiliary text.
2. Keep only incorrect predictions.
3. Keep only cases where the wrong prediction appears in the corrupted auxiliary text.
4. Build DPO pairs:
   - `chosen`: image-grounded ground-truth answer
   - `rejected`: the misleading answer followed from corrupted text

This requires no manual labeling. The label is a weak, high-precision pseudo-label derived from base prediction and auxiliary-text matching.

## Added Utilities

- `make_text_following_span_dpo.py`
  - Builds a DPO dataset from base-model evaluation outputs.
  - Adds `misleading_answer` and `sample_weight` fields.
- `hf_evaluator.py`
  - Adds `--sample_offset` so train-mining and held-out test slices can be disjoint after shuffling.
- `compare_eval_outputs.py`
  - Compares corrected/regressed predictions between two eval JSONL files.
- `measure_text_following_rate.py`
  - Measures how often predictions appear in corrupted auxiliary text.
- `make_hard_weighted_text_following_dpo.py`
  - Builds a DPO dataset that upweights high-confidence corrupted-text-copy errors.
- `make_preservation_sft_dataset.py`
  - Builds match/irrelevant SFT anchor data for preservation regularization.
- `train_dpo_sft_qwen2vl_lora.py`
  - Trains text-following DPO together with an auxiliary preservation SFT loss.

## Main Same-Pool Result

Model: `Qwen/Qwen3-VL-8B-Instruct`

Training data:

- Source eval: `results/qwen3vl8b_base_corrupted_1000.jsonl`
- Extracted text-following rows: 251
- Train/test after DPO dataset split: 225 / 26
- Training: LoRA rank 8, alpha 16, lr `1e-5`, beta `0.1`, 700 DPO steps

DocVQA corrupted 1000:

| Model | Accuracy | Incorrect aux-hit | Corrected / Regressed |
|---|---:|---:|---:|
| Base Qwen3-VL-8B | 0.724 | 90.2% | - |
| Base-error DPO | 0.790 | 84.3% | 67 / 1 |
| Text-following span DPO | 0.895 | 61.0% | 174 / 3 |

Preservation:

| Eval | Accuracy |
|---|---:|
| match 100 | 1.00 |
| irrelevant 100 | 0.96 |

## Disjoint Held-Out Split

To reduce same-pool leakage, we split the shuffled corrupted DocVQA pool:

- Train-mining slice: seed 0, offset 0, 800 examples
- Held-out test slice: seed 0, offset 800, 200 examples

From the train-mining slice:

- Extracted text-following rows: 200
- DPO train/test rows: 180 / 20
- Same DPO hyperparameters as above

Held-out corrupted 200, seed 0:

| Model | Accuracy | Incorrect aux-hit | Corrected / Regressed |
|---|---:|---:|---:|
| Base Qwen3-VL-8B | 0.715 | 89.5% | - |
| Text-following span DPO | 0.900 | 55.0% | 37 / 0 |

We repeated the same protocol with a different shuffle seed:

- Train-mining slice: seed 1, offset 0, 800 examples
- Held-out test slice: seed 1, offset 800, 200 examples
- Extracted text-following rows: 251
- DPO train/test rows: 225 / 26

Held-out corrupted 200, seed 1:

| Model | Accuracy | Incorrect aux-hit | Corrected / Regressed |
|---|---:|---:|---:|
| Base Qwen3-VL-8B | 0.760 | 89.6% | - |
| Text-following span DPO | 0.930 | 50.0% | 34 / 0 |

Preservation on seed 1:

| Eval | Accuracy |
|---|---:|
| match 100 | 0.99 |
| irrelevant 100 | 0.97 |

## DocVQA DPO Hyperparameter Grid

We ran a small 2x2 ablation on the seed-0 disjoint DocVQA split, keeping the same train-mining slice, held-out test slice, learning rate (`1e-5`), 700 DPO steps, batch size, and Qwen3-VL-8B base model. Only DPO beta and LoRA rank changed.

Held-out corrupted 200, seed 0:

| Beta | LoRA rank | Accuracy | Incorrect aux-hit | Corrected / Regressed | match 100 | irrelevant 100 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 8 | 0.900 | 55.0% | 37 / 0 | 1.00 | 0.96 |
| 0.2 | 8 | 0.900 | 55.0% | 37 / 0 | 0.98 | 0.91 |
| 0.1 | 16 | 0.915 | 47.1% | 40 / 0 | 0.98 | 0.91 |
| 0.2 | 16 | 0.915 | 47.1% | 40 / 0 | 0.98 | 0.91 |

The best setting in this grid is LoRA rank 16, with either beta. Increasing rank from 8 to 16 improved corrupted accuracy from 0.900 to 0.915, added three more corrected examples, kept regressions at zero, and reduced incorrect corrupted-text copying from 55.0% to 47.1%. Increasing beta from 0.1 to 0.2 did not change the corrupted metric in either rank setting. Because the r16 runs slightly reduce match/irrelevant preservation relative to the original r8 beta-0.1 run, the next useful check is whether r16 remains better under a larger held-out slice or another seed, rather than treating the 0.015 gain as fully settled.

## Counterfactual and Multi-Condition DPO Checks

We then tested whether adding non-corrupted prompt conditions can further reduce text bias without moving away from DPO. These are DPO-only experiments; they should not be mixed with the GRPO-only results.

### Prompt-augmented counterfactual DPO

The first variant expanded each mined corrupted failure into four prompt conditions: corrupted, no auxiliary text, matching auxiliary text, and irrelevant auxiliary text. However, all rows still used the same preference pair:

- `chosen`: ground-truth answer
- `rejected`: original corrupted-text-following answer

This produced 800 DPO rows from the 200 seed-0 corrupted failures and trained a rank-16, beta-0.1 adapter for 1400 steps.

Held-out corrupted 200, seed 0:

| Model | Accuracy | Incorrect aux-hit | Corrected / Regressed vs base | match 100 | irrelevant 100 |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-VL-8B | 0.715 | 89.5% | - | 0.99 | 0.91 |
| Plain span DPO, r16 beta 0.1 | 0.915 | 47.1% | 40 / 0 | 0.98 | 0.91 |
| Prompt-augmented counterfactual DPO | 0.915 | 47.1% | 40 / 0 | 0.98 | 0.91 |

This tied the plain r16 result. The likely reason is that the augmented prompts did not introduce condition-specific negative behavior; they mostly repeated the same `GT > corrupted-answer` preference under different prompt wrappers.

### Multi-condition error-mined DPO

The second variant mines base-model failures separately under each condition and uses the actual failed prediction as the rejected answer for that condition. This makes the non-corrupted rows condition-specific instead of prompt duplicates.

Dataset: `data/dpo_DocVQA_multicond_error_seed0_train800`

- Total examples: 273
- Train/test split: 245 / 28
- Source counts: corrupted 200, match 7, irrelevant 33, no-text 33
- Training: Qwen3-VL-8B, LoRA rank 16, alpha 32, lr `1e-5`, beta `0.1`, 900 DPO steps

Held-out corrupted 200, seed 0:

| Model | Accuracy | Incorrect aux-hit | Corrected / Regressed vs base | Changed vs plain r16 | match 100 | irrelevant 100 |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-8B | 0.715 | 89.5% | - | - | 0.99 | 0.91 |
| Plain span DPO, r16 beta 0.1 | 0.915 | 47.1% | 40 / 0 | - | 0.98 | 0.91 |
| Multi-condition error-mined DPO | 0.920 | 43.8% | 41 / 0 | +1 corrected / 0 regressed | 0.98 | 0.91 |

The gain is small but directionally useful: it fixes one additional held-out corrupted example relative to plain r16 DPO and further lowers the rate at which remaining wrong answers appear in the corrupted auxiliary text. Because match/no-text failures are sparse on DocVQA, the added signal is limited. The result supports the design principle that counterfactual DPO rows need condition-specific rejected answers, not just condition-specific prompt wrappers.

## VQAv2 Transfer Check

We also ran the same disjoint split protocol on VQAv2 to test whether the method is specific to document images.

- Train-mining slice: seed 0, offset 0, 800 examples
- Held-out test slice: seed 0, offset 800, 200 examples
- Extracted text-following rows: 192
- DPO train/test rows: 172 / 20
- Same Qwen3-VL-8B and DPO hyperparameters as DocVQA

Held-out corrupted 200, VQAv2 seed 0:

| Model | Accuracy | Incorrect aux-hit | Corrected / Regressed |
|---|---:|---:|---:|
| Base Qwen3-VL-8B | 0.545 | 65.9% | - |
| Text-following span DPO | 0.645 | 40.8% | 22 / 2 |

Preservation on VQAv2 seed 0:

| Eval | Accuracy |
|---|---:|
| match 100 | 0.90 |
| irrelevant 100 | 0.75 |

We then tried a more conservative DPO recipe on the same mined data (`lr=5e-6`, 300 steps instead of `lr=1e-5`, 700 steps):

| Model | Accuracy | Incorrect aux-hit | Corrected / Regressed | match 100 | irrelevant 100 |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-VL-8B | 0.545 | 65.9% | - | 0.92 | 0.77 |
| Span DPO, 700 steps | 0.645 | 40.8% | 22 / 2 | 0.90 | 0.75 |
| Span DPO, 300 steps, lr 5e-6 | 0.555 | 65.2% | 2 / 0 | 0.92 | 0.77 |
| Hard-weighted span DPO, lr 7e-6, 700 steps | 0.600 | 55.0% | 12 / 1 | 0.90 | 0.77 |
| Span DPO + preservation SFT, w=0.2 | 0.635 | 42.5% | 20 / 2 | 0.90 | 0.75 |

This is a positive transfer signal but weaker than DocVQA. The conservative recipe is more stable but barely changes the corrupted behavior, so simple step/lr reduction is not enough. A hard-weighted DPO variant also underperformed the plain span DPO baseline. It used the same 192 mined VQAv2 text-following rows but assigned larger weights to high-confidence corrupted-text-copy errors (`weight_min/max/avg = 1.50/2.25/2.21`) and trained with `lr=7e-6` for 700 steps. The final checkpoint reached 0.600 corrupted accuracy and 55.0% incorrect aux-hit; intermediate checkpoint sweep was 0.545/0.550/0.550/0.565/0.600/0.615 at steps 100/200/300/400/500/600. So the current weighting heuristic does not improve over plain DPO.

We also tried explicit preservation regularization by mixing DPO with an auxiliary SFT loss on VQAv2 match/irrelevant samples. With `sft_weight=0.2`, this did not materially improve preservation versus the 700-step DPO baseline: match and irrelevant stayed at 0.90/0.75, while corrupted accuracy decreased slightly from 0.645 to 0.635 and incorrect aux-hit increased from 40.8% to 42.5%. This suggests both naive stronger weighting and the current SFT anchor are too weak or too loosely aligned with the failure mode. Better next variants are GRPO with a reward that explicitly separates image-grounded correction from generic answer drift, or a more selective DPO dataset rather than simply scaling all high-copy pseudo-labels.

## Interpretation

The result suggests that a large fraction of corrupted-context failures are conflict-resolution failures: the model follows a plausible textual answer even when the image supports a different answer. Filtering training pairs to this high-precision failure type gives a cleaner preference signal than using all base errors.

The seed-1 repeat strengthens the DPO-only evidence: accuracy improved on a disjoint held-out slice while the rate of incorrect predictions copied from corrupted text dropped sharply. The DocVQA 2x2 hyperparameter grid suggests extra LoRA capacity is more useful than increasing beta: r16 improves corrupted accuracy and text-bias reduction, while beta 0.2 is effectively tied with beta 0.1. The prompt-augmented counterfactual check was a negative result because it duplicated the same rejected answer across conditions. The multi-condition error-mined variant gives a small additional gain, suggesting that condition-specific rejected answers are more useful than prompt augmentation alone, but the effect is currently modest because non-corrupted failures are rare. The VQAv2 transfer check shows the same direction but weaker preservation. The conservative ablation, hard-weighted ablation, and first DPO+SFT preservation run did not improve over pure DPO, so broad generalization should not be claimed yet. GRPO-only, DPO-only, and DPO-to-GRPO experiments should remain reported separately; SFT should be treated as a negative ablation rather than a main method.

## Example Commands

Build text-following span DPO data:

```bash
python our_codes/make_text_following_span_dpo.py \
  --eval_file results/qwen3vl8b_base_corrupted_1000.jsonl \
  --out_dir data/dpo_DocVQA_text_follow_span_base_errors_corrupted_1000 \
  --subset DocVQA \
  --text_type corrupted \
  --seed 0
```

Train DPO:

```bash
CUDA_VISIBLE_DEVICES=0 python our_codes/train_dpo_qwen2vl_lora.py \
  --dataset_dir data/dpo_DocVQA_text_follow_span_base_errors_corrupted_1000 \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --output_dir checkpoints/qwen3vl8b_dpo_text_follow_span_r8_lr1e5_700_ada \
  --max_steps 700 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 1e-5 \
  --beta 0.1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --max_pixels 401408
```

Evaluate a held-out shuffled slice:

```bash
python hf_evaluator.py \
  --ds_name dal-289/word_or_vision \
  --subset DocVQA \
  --model_type adapter \
  --model_name checkpoints/qwen3vl8b_dpo_span_split_train800_r8_lr1e5_700_ada_merged \
  --input_type text+image \
  --text_type corrupted \
  --max_samples 200 \
  --sample_offset 800 \
  --seed 0 \
  --out_file results/qwen3vl8b_dpo_span_split_train800_corrupted_heldout200.jsonl
```


Build hard-weighted VQAv2 text-following DPO data:

```bash
python our_codes/make_hard_weighted_text_following_dpo.py \
  --eval_file results/qwen3vl8b_base_vqav2_corrupted_seed0_train800.jsonl \
  --out_dir data/dpo_VQAv2_text_follow_hard_weighted_seed0_train800 \
  --subset VQAv2 \
  --text_type corrupted \
  --weight_mode hard \
  --seed 0
```


Train VQAv2 DPO with preservation SFT anchors:

```bash
python our_codes/make_preservation_sft_dataset.py \
  --out_dir data/sft_VQAv2_preserve_match_irrelevant_seed0_200each \
  --subset VQAv2 \
  --max_per_type 200 \
  --seed 0

CUDA_VISIBLE_DEVICES=0 python our_codes/train_dpo_sft_qwen2vl_lora.py \
  --dataset_dir data/dpo_VQAv2_text_follow_span_seed0_train800 \
  --sft_dataset_dir data/sft_VQAv2_preserve_match_irrelevant_seed0_200each \
  --sft_weight 0.2 \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --output_dir checkpoints/qwen3vl8b_dpo_sft_vqav2_seed0_train800_r8_lr1e5_700_sftw02_ada \
  --max_steps 700 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 1e-5 \
  --beta 0.1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --max_pixels 401408
```
