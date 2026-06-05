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

Held-out corrupted 200:

| Model | Accuracy | Incorrect aux-hit | Corrected / Regressed |
|---|---:|---:|---:|
| Base Qwen3-VL-8B | 0.715 | 89.5% | - |
| Text-following span DPO | 0.900 | 55.0% | 37 / 0 |

## Interpretation

The result suggests that a large fraction of corrupted-context failures are conflict-resolution failures: the model follows a plausible textual answer even when the image supports a different answer. Filtering training pairs to this high-precision failure type gives a cleaner preference signal than using all base errors.

The most important next validation is to repeat the same split protocol across additional seeds and other subsets such as VQAv2.

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
