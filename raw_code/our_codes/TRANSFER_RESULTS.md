# Three-Model Transfer Results

This note summarizes the follow-up transfer checks on RTX 6000 Ada GPUs only. The goal was to see whether the text-following span DPO story transfers beyond Qwen3-VL-8B.

## Setup

Models:

- `Qwen/Qwen2-VL-7B-Instruct`
- `llava-hf/llava-v1.6-vicuna-7b-hf`
- `llava-hf/llava-v1.6-vicuna-13b-hf`

Shared protocol:

- Dataset: `dal-289/word_or_vision`, `DocVQA`
- Train-mining split: seed 0, offset 0, 800 corrupted examples
- Held-out split: seed 0, offset 800, 200 corrupted examples
- Preservation checks: match 100, irrelevant 100
- GPUs: RTX 6000 Ada only. H100 was not used.

## Qwen2-VL-7B

Qwen2-VL-7B used the same Qwen DPO trainer as Qwen3-VL. The mined text-following span DPO data was built from the model's own corrupted train800 failures.

| Model | Corrupted Acc | Incorrect aux-hit | Corrected / Regressed | Match 100 | Irrelevant 100 |
|---|---:|---:|---:|---:|---:|
| Qwen2-VL-7B base | 0.570 | 93.0% | - | - | - |
| Qwen2-VL-7B span DPO r16 beta 0.1 | 0.775 | 86.7% | 41 / 0 | 0.98 | 0.94 |

Interpretation:

- This is a strong positive transfer result.
- Accuracy improves by +0.205 on the held-out corrupted split.
- The model fixes 41 examples without any regressions against the base predictions.
- The incorrect aux-hit rate remains high, but absolute accuracy improves substantially.
- This supports the claim that text-following span DPO is not specific to Qwen3-VL-8B.

## LLaVA-Next-7B

LLaVA-Next-7B loaded and evaluated successfully, but its DocVQA corrupted base performance was very low. The model still produced many corrupted-text-copy errors, so we ran a conservative DPO pilot using a memory-saving LLaVA trainer that precomputes reference log-probs.

Mined data:

- Train rows: 579
- Test rows: 65

| Model | Corrupted Acc | Incorrect aux-hit | Corrected / Regressed | Match 100 | Irrelevant 100 |
|---|---:|---:|---:|---:|---:|
| LLaVA-Next-7B base | 0.105 | 90.5% | - | - | - |
| LLaVA-Next-7B span DPO r16 beta 0.1, 300 steps | 0.100 | 91.7% | 1 / 2 | 0.95 | 0.59 |

Interpretation:

- This is a negative DPO result.
- The base model is already poorly aligned with DocVQA under the current prompt/template.
- DPO does not improve corrupted accuracy and slightly increases corrupted-text copying among remaining errors.
- It also causes two regressions and only one correction.
- The result suggests that LLaVA-Next needs prompt/template or task adaptation before this DPO recipe becomes meaningful.

## LLaVA-Next-13B

LLaVA-Next-13B base/mining completed successfully. We then ran the same conservative DPO pilot used for LLaVA-Next-7B, because this comparison is useful for separating "DPO does not transfer to LLaVA under the current setup" from "7B was simply too small."

Mined data:

- Train rows: 585
- Test rows: 66

| Model | Corrupted Acc | Incorrect aux-hit | Corrected / Regressed | Match 100 | Irrelevant 100 |
|---|---:|---:|---:|---:|---:|
| LLaVA-Next-13B base | 0.115 | 92.7% | - | - | - |
| LLaVA-Next-13B span DPO r16 beta 0.1, 300 steps | 0.130 | 95.4% | 3 / 0 | 0.97 | 0.62 |

Interpretation:

- The 13B base performance is only slightly above LLaVA-Next-7B and still far below Qwen2/Qwen3.
- DPO gives a tiny absolute accuracy gain (+0.015) and no regressions against the base predictions, but this is not a convincing text-bias reduction result.
- The incorrect aux-hit rate increases from 92.7% to 95.4%, meaning the remaining wrong answers are even more likely to appear in corrupted auxiliary text.
- This supports the boundary-condition interpretation: LLaVA-Next is poorly aligned with DocVQA under the current prompt/template, so DPO on mined text-copy failures is not enough by itself.
- The better next step for LLaVA would be prompt/template diagnosis or a short supervised/task-adaptation check before applying this DPO recipe.

## Overall Takeaway

The transfer check strengthens the Qwen-family result but weakens the case for claiming architecture-agnostic behavior.

- Qwen2-VL-7B: strong positive transfer.
- LLaVA-Next-7B: negative under the current prompt/template.
- LLaVA-Next-13B: tiny accuracy gain from DPO, but no evidence of text-bias reduction.

The clean story is:

> Text-following span DPO transfers well within the Qwen-VL family, where the base model already has reasonable DocVQA ability. For LLaVA-Next, the current prompt/template produces very low DocVQA accuracy, so DPO on mined text-copy failures is not enough by itself. This suggests the method depends on a base VLM that can already read and answer the task reasonably well; DPO then corrects conflict-resolution behavior rather than teaching the task from scratch.
