# Ablation: Text Bias 완화 실험의 추가 검증

이 문서는 `Overall_Result.md`에 넣기에는 다소 세부적인 ablation 및 추가 검증 결과를 정리한다. 핵심 메시지는 전체 결과 문서에 두고, 여기서는 실험 조건, 비교 기준, 해석상의 caveat를 남긴다.

## 1. TextVQA pilot: 원 benchmark 밖 일반화 검증

### 목적

DocVQA/VQAv2 중심으로 확인한 text-following span DPO가 원래 실험과 다른 VQA dataset에서도 효과를 보이는지 확인하기 위해 TextVQA validation split에서 200개 pilot을 만들었다.

TextVQA는 이미지 안의 scene text를 읽어 답하는 dataset이다. 따라서 DocVQA처럼 문서 이미지와 보조 텍스트의 충돌을 다루는 setting과 완전히 같지는 않다. 이 실험은 "모든 VQA dataset에서 성능이 크게 오르는가"를 주장하기 위한 것이 아니라, 원 benchmark 밖에서 효과가 어느 정도 남는지 보는 sanity check에 가깝다.

### Corrupted text 생성 방식

1. TextVQA validation split에서 answer가 yes/no, empty, unknown류가 아닌 샘플만 사용했다.
2. TextVQA의 여러 human answer 중 majority-normalized answer를 canonical answer로 잡았다.
3. 같은 이미지의 OCR token 중 canonical answer 및 human answer들과 겹치지 않는 token을 wrong auxiliary text로 선택했다.
4. corrupted 조건에는 wrong token을 넣고, match 조건에는 canonical answer를 넣었다.
5. auxiliary text의 filler OCR token도 정답 후보들과 겹치지 않도록 필터링했다.

이 방식은 TextVQA의 실제 OCR token을 이용하므로 완전히 무작위인 wrong text보다 자연스럽다. 다만 TextVQA answer normalization 특성상 `gt_in_aux_text_rate_all`이 0으로 완전히 떨어지지는 않았다. 엄격 필터 이후에도 5% 수준의 overlap이 남았다.

### 비교 모델

| 항목 | 모델 |
|---|---|
| Baseline | `Qwen/Qwen3-VL-8B-Instruct` |
| DPO | DocVQA text-following span DPO, LoRA r16, beta 0.1 |

DPO 모델은 TextVQA로 추가 학습하지 않았다. 즉 이 실험은 TextVQA-specific tuning이 아니라 DocVQA에서 학습한 DPO가 TextVQA corrupted setting으로 얼마나 전이되는지 보는 실험이다.

### 결과

| TextVQA pilot, n=200 | Corrupted soft acc | Corrupted exact acc | Match soft acc | Match exact acc | Incorrect aux-hit |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-VL-8B | 0.818 | 0.870 | 0.861 | 0.910 | 11.5% |
| DocVQA span DPO | 0.823 | 0.875 | 0.861 | 0.910 | 12.0% |

Prediction-level comparison은 다음과 같다.

| Condition | Changed | Corrected | Regressed |
|---|---:|---:|---:|
| Corrupted | 6 / 200 | 1 | 0 |
| Match | 10 / 200 | 1 | 1 |

### 해석

TextVQA pilot에서는 DPO의 이득이 매우 작았다. corrupted soft accuracy는 0.818에서 0.823으로 0.0045만 올랐고, match 조건은 변하지 않았다. exact 기준으로도 corrupted에서 0.870에서 0.875로 1개 샘플만 개선되었다.

이 결과는 다음처럼 해석하는 것이 가장 안전하다.

> DocVQA에서 학습한 text-following span DPO가 TextVQA로 강하게 일반화되었다고 보기는 어렵다. 다만 TextVQA 성능을 해치지는 않았고, corrupted 조건에서 아주 작은 개선만 보였다.

왜 개선 폭이 작았는지는 task 성격 차이로 설명할 수 있다. TextVQA는 주로 이미지 속 글자를 읽는 능력이 중요하다. 반면 우리가 학습한 DPO는 DocVQA에서 보조 텍스트와 이미지 evidence가 충돌할 때, corrupted auxiliary text를 그대로 따라가지 않는 선택을 학습한 것이다. 즉 둘 다 text와 관련되어 있지만, TextVQA는 OCR reading 자체가 핵심이고, DocVQA corrupted setting은 conflict resolution이 핵심이다.

따라서 TextVQA pilot은 연구 주장에 다음 경계 조건을 추가한다.

> 이 방법은 일반 VQA 성능을 무조건 끌어올리는 범용 tuning이라기보다, auxiliary text와 visual evidence가 충돌하는 상황에서 특히 잘 작동하는 text-bias 완화 방법이다.

## 2. 재현 스크립트

TextVQA pilot 관련 코드는 다음 파일에 있다.

| 파일 | 역할 |
|---|---|
| `our_codes/make_textvqa_corrupted_pilot.py` | TextVQA validation에서 corrupted/match local dataset 생성 |
| `our_codes/run_textvqa_pilot_20260609.sh` | baseline, DPO, match/corrupted 평가 및 비교 실행 |
| `hf_evaluator.py` | local `Dataset.save_to_disk()` 경로를 평가할 수 있도록 `load_from_disk` 지원 추가 |

주요 산출물은 다음 경로에 저장된다.

| 경로 | 내용 |
|---|---|
| `data/textvqa_corruption_pilot_seed0_200/` | 생성된 TextVQA corrupted/match local dataset |
| `results/textvqa_pilot/` | baseline/DPO 평가 결과 및 comparison report |

