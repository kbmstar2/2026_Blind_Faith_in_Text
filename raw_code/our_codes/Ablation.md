# Ablation: Text Bias 완화 실험의 추가 검증

이 문서는 `Overall_Result.md`에 넣기에는 다소 세부적인 ablation 및 추가 검증 결과를 정리한다. 핵심 메시지는 전체 결과 문서에 두고, 여기서는 실험 조건, 비교 기준, 해석상의 caveat를 남긴다.

## 1. TextVQA pilot: 원 benchmark 밖 일반화 검증

### 목적

DocVQA/VQAv2 중심으로 확인한 text-following span DPO가 원래 실험과 다른 VQA dataset에서도 효과를 보이는지 확인하기 위해 TextVQA validation split에서 200개 pilot을 만들었다.

TextVQA는 이미지 안의 scene text를 읽어 답하는 dataset이다. 따라서 DocVQA처럼 문서 이미지와 보조 텍스트의 충돌을 다루는 setting과 완전히 같지는 않다. 이 실험은 "모든 VQA dataset에서 성능이 크게 오르는가"를 주장하기 위한 것이 아니라, 원 benchmark 밖에서 효과가 어느 정도 남는지 보는 sanity check에 가깝다.

### Corrupted text 생성 방식 A: same-image OCR distractor

1. TextVQA validation split에서 answer가 yes/no, empty, unknown류가 아닌 샘플만 사용했다.
2. TextVQA의 여러 human answer 중 majority-normalized answer를 canonical answer로 잡았다.
3. 같은 이미지의 OCR token 중 canonical answer 및 human answer들과 겹치지 않는 token을 wrong auxiliary text로 선택했다.
4. corrupted 조건에는 wrong token을 넣고, match 조건에는 canonical answer를 넣었다.
5. auxiliary text의 filler OCR token도 정답 후보들과 겹치지 않도록 필터링했다.

이 방식은 TextVQA의 실제 OCR token을 이용하므로 완전히 무작위인 wrong text보다 자연스럽다. 다만 같은 이미지에서 가져온 OCR token이기 때문에, "보조 텍스트가 이미지와 완전히 무관하다"는 corruption은 아니다. 더 정확히는 이미지 안에 실제로 존재하는 다른 OCR token을 distractor로 넣은 setting이다.

TextVQA answer normalization 특성상 `gt_in_aux_text_rate_all`이 0으로 완전히 떨어지지는 않았다. 엄격 필터 이후에도 5% 수준의 overlap이 남았다.

### 비교 모델

| 항목 | 모델 |
|---|---|
| Baseline | `Qwen/Qwen3-VL-8B-Instruct` |
| DPO | DocVQA text-following span DPO, LoRA r16, beta 0.1 |

DPO 모델은 TextVQA로 추가 학습하지 않았다. 즉 이 실험은 TextVQA-specific tuning이 아니라 DocVQA에서 학습한 DPO가 TextVQA corrupted setting으로 얼마나 전이되는지 보는 실험이다.

### 결과 A: same-image OCR distractor

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

### Corrupted text 생성 방식 B: cross-sample answer corruption

same-image OCR distractor는 자연스럽지만, auxiliary text가 이미지와 완전히 무관한 것은 아니다. 그래서 더 강한 corruption으로 `cross-sample answer corruption`을 추가했다.

절차는 다음과 같다.

1. 현재 샘플의 image/question/GT answer는 그대로 둔다.
2. 다른 TextVQA 샘플의 majority answer를 가져온다.
3. 이 answer가 현재 샘플의 GT answer들과 겹치지 않는 경우에만 wrong auxiliary text로 사용한다.
4. match 조건은 기존처럼 현재 샘플의 canonical answer를 auxiliary text에 넣는다.

즉 이 setting은 "같은 이미지 안의 다른 OCR token"이 아니라, 아예 다른 샘플에서 온 답을 보조 텍스트로 붙인다. 사용자가 말한 "아예 관계없는 다른 text를 붙이는 corrupted text"에 더 가까운 설정이다.

### 결과 B: cross-sample answer corruption

| TextVQA cross-sample pilot, n=200 | Corrupted soft acc | Corrupted exact acc | Match soft acc | Match exact acc | Incorrect aux-hit |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-VL-8B | 0.819 | 0.875 | 0.861 | 0.910 | 8.0% |
| DocVQA span DPO | 0.823 | 0.880 | 0.861 | 0.910 | 8.3% |

Prediction-level comparison은 다음과 같다.

| Condition | Changed | Corrected | Regressed |
|---|---:|---:|---:|
| Corrupted | 8 / 200 | 1 | 0 |
| Match | 10 / 200 | 1 | 1 |

### Same-image vs cross-sample 해석

두 TextVQA corruption 모두 결과가 거의 비슷했다.

| Corruption | Base corrupted soft | DPO corrupted soft | Delta | Base exact | DPO exact |
|---|---:|---:|---:|---:|---:|
| Same-image OCR distractor | 0.818 | 0.823 | +0.005 | 0.870 | 0.875 |
| Cross-sample answer | 0.819 | 0.823 | +0.004 | 0.875 | 0.880 |

이 결과는 중요한 단서를 준다. TextVQA에서는 보조 텍스트를 같은 이미지 OCR에서 가져오든, 다른 샘플 answer에서 가져오든, base Qwen3가 auxiliary text를 많이 따라가지 않았다. 실제로 incorrect aux-hit도 8.0~11.5% 수준으로 낮았다. DocVQA main experiment에서 base incorrect aux-hit가 89.5%였던 것과 매우 다르다.

따라서 TextVQA에서 DPO 이득이 작은 이유는 corruption이 약해서라기보다, 이 TextVQA prompt/dataset setting에서 base model이 애초에 auxiliary text shortcut을 강하게 타지 않았기 때문일 가능성이 크다. TextVQA는 scene text를 직접 읽는 문제가 많아서, 모델이 보조 텍스트보다 이미지 OCR evidence를 더 많이 사용했을 수 있다.

결론적으로 TextVQA pilot은 다음처럼 정리할 수 있다.

> TextVQA에서는 cross-sample unrelated text를 붙여도 text-following failure가 많이 발생하지 않았다. 따라서 DocVQA DPO의 효과도 작게 나타났다. 이 결과는 우리의 방법이 "모든 VQA dataset에서 성능을 올리는 방법"이라기보다, base VLM이 실제로 corrupted auxiliary text를 따라가는 conflict setting에서 특히 효과적이라는 해석을 강화한다.

### Corrupted text 생성 방식 C: question-type matched answer hint

Cross-sample answer corruption도 conflict가 약했을 가능성이 있다. 아무 answer나 붙이면 질문과 맞지 않아 모델이 쉽게 무시할 수 있기 때문이다. 그래서 세 번째로 `question-type matched answer hint`를 만들었다.

이 방식은 다음 두 가지를 바꾼다.

1. Wrong answer를 고를 때 질문 타입을 맞춘다. 예를 들어 number 질문에는 다른 number, color 질문에는 다른 color, brand 질문에는 다른 brand answer를 우선 사용한다.
2. Auxiliary text 문구를 단순 OCR list가 아니라 answer hint에 가깝게 만든다.

Prompt에 들어간 auxiliary text 형식은 다음과 같다.

```text
A possibly noisy scene-text QA parser predicts the answer is: "{wrong_answer}".
```

즉 이전 두 실험보다 더 직접적으로 "이 답일 수 있다"는 보조 텍스트를 제공한다. 이 setting은 TextVQA에서 conflict를 최대한 강하게 재현하려는 시도다.

### 결과 C: question-type matched answer hint

| TextVQA type-matched pilot, n=200 | Corrupted soft acc | Corrupted exact acc | Match soft acc | Match exact acc | Incorrect aux-hit |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-VL-8B | 0.821 | 0.870 | 0.911 | 0.945 | 11.5% |
| DocVQA span DPO | 0.827 | 0.880 | 0.904 | 0.940 | 8.3% |

Prediction-level comparison은 다음과 같다.

| Condition | Changed | Corrected | Regressed |
|---|---:|---:|---:|
| Corrupted | 13 / 200 | 2 | 0 |
| Match | 8 / 200 | 1 | 2 |

### 세 TextVQA corruption 비교

| Corruption | Base corrupted soft | DPO corrupted soft | Delta | Base exact | DPO exact | Base incorrect aux-hit |
|---|---:|---:|---:|---:|---:|---:|
| Same-image OCR distractor | 0.818 | 0.823 | +0.005 | 0.870 | 0.875 | 11.5% |
| Cross-sample answer | 0.819 | 0.823 | +0.004 | 0.875 | 0.880 | 8.0% |
| Question-type matched answer hint | 0.821 | 0.827 | +0.006 | 0.870 | 0.880 | 11.5% |

Question-type matched answer hint는 앞선 두 방식보다 더 직접적인 conflict를 만들었지만, 여전히 base Qwen3가 corrupted answer hint를 많이 따라가지는 않았다. Base incorrect aux-hit는 11.5% 수준으로, DocVQA main experiment의 89.5%와 비교하면 매우 낮다.

흥미로운 점은 match condition이다. Match에서는 auxiliary text가 정답을 직접 answer hint로 제공하므로 base exact accuracy가 0.945까지 올라간다. 즉 모델이 보조 텍스트를 완전히 무시하는 것은 아니다. 다만 보조 텍스트가 틀렸을 때는 이를 정답으로 강하게 따라가지 않는다. 이 차이는 TextVQA에서는 Qwen3가 이미지 안의 OCR evidence를 직접 확인하는 능력이 강하고, corrupted answer hint와 이미지 evidence가 충돌하면 이미지 쪽을 더 신뢰할 수 있음을 시사한다.

따라서 TextVQA에서 conflict setting 재현이 어려웠던 이유는 다음처럼 정리할 수 있다.

> TextVQA-Qwen3 조합에서는 auxiliary text가 정답일 때는 도움이 되지만, auxiliary text가 이미지와 충돌할 때는 모델이 이를 많이 따라가지 않는다. 즉 우리가 원하는 text-following failure 자체가 적게 발생한다.

이 결과는 우리의 방법론을 약화한다기보다, 적용 조건을 더 명확히 한다. Text-following span DPO는 base model이 실제로 corrupted text를 따라가는 setting에서 효과가 크다. TextVQA처럼 base model이 visual/OCR evidence를 이미 강하게 사용하는 setting에서는 학습할 failure가 적기 때문에 개선 폭도 작다.

### Qwen2-VL / LLaVA-Next에서도 같은가

Qwen3-VL-8B만 특이하게 corrupted answer hint를 잘 무시하는지 확인하기 위해, 같은 `question-type matched answer hint` dataset을 Qwen2-VL-7B, LLaVA-Next-7B, LLaVA-Next-13B에도 적용했다. 각 모델은 DocVQA text-following span DPO checkpoint와 base model을 비교했다.

| Model | Base corrupted soft | DPO corrupted soft | Base match soft | DPO match soft | Base incorrect aux-hit | DPO incorrect aux-hit |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 0.821 | 0.827 | 0.911 | 0.904 | 11.5% | 8.3% |
| Qwen2-VL-7B | 0.832 | 0.845 | 0.870 | 0.875 | 23.1% | 8.3% |
| LLaVA-Next-7B | 0.535 | 0.443 | 0.863 | 0.874 | 50.0% | 62.0% |
| LLaVA-Next-13B | 0.612 | 0.602 | 0.867 | 0.848 | 35.7% | 28.8% |

Exact-match 기준의 prediction-level 변화는 다음과 같다.

| Model | Corrupted exact: base -> DPO | Corrected / Regressed | Match exact: base -> DPO | Corrected / Regressed |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | 0.870 -> 0.880 | 2 / 0 | 0.945 -> 0.940 | 1 / 2 |
| Qwen2-VL-7B | 0.870 -> 0.880 | 2 / 0 | 0.900 -> 0.905 | 1 / 0 |
| LLaVA-Next-7B | 0.560 -> 0.460 | 0 / 20 | 0.895 -> 0.910 | 3 / 0 |
| LLaVA-Next-13B | 0.650 -> 0.635 | 4 / 7 | 0.905 -> 0.890 | 1 / 4 |

이 결과는 Qwen 계열과 LLaVA 계열이 TextVQA conflict setting에서 다르게 반응한다는 것을 보여준다.

Qwen2-VL-7B는 Qwen3보다 base incorrect aux-hit가 높다. 즉 Qwen2는 corrupted answer hint를 더 자주 따라간다. 하지만 DocVQA DPO 이후에는 incorrect aux-hit가 23.1%에서 8.3%로 내려갔고, corrupted accuracy도 소폭 개선되었다. 이는 Qwen 계열에서는 DocVQA에서 학습한 text-following DPO가 TextVQA conflict에도 어느 정도 전이될 수 있음을 보여준다.

반면 LLaVA-Next-7B는 conflict가 매우 잘 걸린다. Base incorrect aux-hit가 50.0%로 높고, match condition에서는 0.863으로 준수한 성능을 낸다. 즉 LLaVA-7B는 보조 텍스트가 정답이면 잘 활용하지만, corrupted answer hint가 들어오면 상당히 흔들린다. 더 중요한 점은 DocVQA DPO 이후 corrupted 성능이 0.535에서 0.443으로 크게 떨어지고, incorrect aux-hit도 50.0%에서 62.0%로 올라간다는 것이다. 실제 prediction 비교에서도 corrected는 0개, regressed는 20개였다.

LLaVA-Next-13B도 base incorrect aux-hit가 35.7%로 Qwen보다 높다. 다만 DPO 이후 aux-hit는 28.8%로 줄었지만, accuracy는 0.612에서 0.602로 소폭 하락했다. 즉 13B는 7B보다 덜 무너지지만, DocVQA DPO가 TextVQA conflict를 안정적으로 개선한다고 보기는 어렵다.

따라서 TextVQA conflict 재현에 대한 결론은 다음처럼 정리할 수 있다.

> Conflict setting 자체는 LLaVA-Next에서 훨씬 잘 재현된다. 특히 LLaVA-7B는 corrupted answer hint를 강하게 따라간다. 다만 DocVQA에서 학습한 DPO가 이 TextVQA conflict를 바로 해결하지는 못했고, 오히려 LLaVA-7B에서는 text-following을 악화했다.

이 결과는 중요한 방법론적 함의를 준다. Text bias 완화는 model family와 task format에 민감하다. Qwen 계열에서는 DocVQA DPO가 TextVQA로 약하게나마 전이되지만, LLaVA 계열에서는 같은 DPO가 다른 dataset의 answer-hint conflict를 안정적으로 해결하지 못한다. 따라서 LLaVA에서 TextVQA conflict를 줄이려면 DocVQA DPO를 그대로 가져오기보다, LLaVA 자체의 TextVQA conflict errors를 mining해서 별도의 DPO pair를 구성하는 편이 더 타당하다.

## 2. 재현 스크립트

TextVQA pilot 관련 코드는 다음 파일에 있다.

| 파일 | 역할 |
|---|---|
| `our_codes/make_textvqa_corrupted_pilot.py` | TextVQA validation에서 corrupted/match local dataset 생성 |
| `our_codes/run_textvqa_pilot_20260609.sh` | baseline, DPO, match/corrupted 평가 및 비교 실행 |
| `our_codes/run_textvqa_cross_sample_pilot_20260609.sh` | cross-sample answer corruption 평가 실행 |
| `our_codes/run_textvqa_type_matched_pilot_20260609.sh` | question-type matched answer hint 평가 실행 |
| `our_codes/run_textvqa_type_matched_transfer_models_20260609.sh` | Qwen2-VL/LLaVA-Next transfer model 평가 실행 |
| `hf_evaluator.py` | local `Dataset.save_to_disk()` 경로를 평가할 수 있도록 `load_from_disk` 지원 추가 |

주요 산출물은 다음 경로에 저장된다.

| 경로 | 내용 |
|---|---|
| `data/textvqa_corruption_pilot_seed0_200/` | 생성된 TextVQA corrupted/match local dataset |
| `results/textvqa_pilot/` | baseline/DPO 평가 결과 및 comparison report |
| `data/textvqa_cross_sample_corruption_pilot_seed0_200/` | cross-sample corrupted/match local dataset |
| `results/textvqa_cross_sample_pilot/` | cross-sample baseline/DPO 평가 결과 및 comparison report |
| `data/textvqa_type_matched_corruption_pilot_seed0_200/` | question-type matched corrupted/match local dataset |
| `results/textvqa_type_matched_pilot/` | question-type matched baseline/DPO 평가 결과 및 comparison report |
| `results/textvqa_type_matched_transfer/` | Qwen2-VL/LLaVA-Next TextVQA conflict 평가 결과 |
