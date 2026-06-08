# Overall Result: VLM의 Text Bias 문제와 DPO 기반 완화 실험 정리

## 1. 왜 이 문제가 중요한가

Vision-Language Model(VLM)은 이미지와 텍스트를 함께 입력으로 받습니다. 문서 이해, OCR 기반 질의응답, 차트/영수증/서류 분석처럼 실제 응용에서는 이미지 안의 시각 정보와 함께 OCR text, caption, retrieved text 같은 보조 텍스트가 자주 들어갑니다.

문제는 이 보조 텍스트가 항상 맞지 않다는 점입니다. OCR이 틀릴 수도 있고, retrieval 결과가 섞일 수도 있고, 공격자가 의도적으로 그럴듯한 잘못된 텍스트를 넣을 수도 있습니다. 이때 VLM이 이미지를 직접 확인하지 않고 보조 텍스트에 있는 답을 그대로 믿으면, 모델은 겉으로는 멀티모달 모델처럼 보이지만 실제로는 텍스트 shortcut에 의존하게 됩니다.

따라서 text bias는 단순히 "텍스트가 있으면 성능이 조금 흔들린다"는 문제가 아닙니다. 더 정확히는 다음 문제입니다.

> 이미지와 텍스트가 서로 충돌할 때, VLM이 무엇을 근거로 답을 선택하는가?

우리가 다루는 문제는 바로 이 충돌 상황입니다. 이미지에는 정답이 있고, corrupted auxiliary text에는 그럴듯하지만 틀린 답이 들어 있습니다. 이때 모델이 corrupted text를 따라가면, 모델은 이미지를 못 본 것이 아니라 이미지보다 텍스트를 더 믿은 것입니다.

## 2. 이 연구의 출발점

기존 `Words or Vision: Do Vision-Language Models Have Blind Faith in Text?` 논문은 VLM이 visual evidence보다 text evidence를 과하게 신뢰할 수 있다는 문제를 제기합니다. 우리는 여기서 한 단계 더 나아가고자 했습니다.

단순히 "text bias가 있다"를 다시 확인하는 것이 아니라, 다음 두 질문에 답하고자 했습니다.

1. Text bias가 실제 예측에서 어떤 형태로 나타나는가?
2. 이 오류를 사람이 직접 라벨링하지 않고 학습 신호로 바꿀 수 있는가?

이를 위해 우리는 text bias를 다음처럼 더 구체적인 오류로 보았습니다.

> 모델이 틀린 답을 냈고, 그 틀린 답이 corrupted auxiliary text 안에 실제로 등장한다면, 이것은 text-following error일 가능성이 높다.

이 정의의 장점은 명확합니다. 사람이 직접 "이 샘플은 text bias다"라고 나누지 않아도 됩니다. 모델의 오답과 corrupted text를 비교하면, corrupted text를 따라간 실패 사례를 자동으로 골라낼 수 있습니다.

## 3. 우리가 제안한 핵심 아이디어

우리가 채택한 핵심 방법은 `text-following span DPO`입니다.

절차는 간단합니다.

1. Base VLM을 corrupted text 조건에서 평가합니다.
2. 모델이 틀린 샘플만 모읍니다.
3. 그중 오답이 corrupted text 안에 실제로 등장하는 경우를 text-following error로 잡습니다.
4. 이 샘플을 DPO preference pair로 바꿉니다.

Preference pair는 다음 구조입니다.

| 항목 | 의미 |
|---|---|
| chosen | 이미지에 근거한 정답 |
| rejected | corrupted text를 따라간 오답 |

즉 모델에게 다음 선호를 학습시킵니다.

> corrupted text에 있는 그럴듯한 오답보다, 이미지에 근거한 정답을 더 선호해야 한다.

이 점이 중요합니다. 우리는 VLM에게 단순히 정답을 외우게 한 것이 아닙니다. 모델이 실제로 유혹당했던 corrupted-text answer를 rejected로 넣어서, "무엇을 믿지 말아야 하는지"까지 알려준 것입니다.

## 4. 왜 DPO인가

이 연구에서 DPO를 쓴 이유는 "RL이 좋아 보여서"가 아닙니다. 문제 구조가 DPO와 잘 맞기 때문입니다.

Text bias는 단순한 지식 부족 문제가 아닙니다. 이미지와 텍스트가 충돌할 때, 모델이 둘 중 어느 쪽을 더 믿을지 잘못 고르는 문제입니다. 다시 말해 이것은 정답 생성 문제이면서 동시에 선호 선택 문제입니다.

SFT는 보통 다음을 가르칩니다.

> 이 입력에서는 이 정답을 출력하라.

DPO는 다음을 가르칩니다.

> 이 답이 저 답보다 낫다. 특히 저 답은 그럴듯해 보여도 선택하면 안 된다.

우리 setting에서는 이 비교 관계가 매우 깨끗하게 만들어집니다.

- 좋은 답: 이미지 기반 GT answer
- 피해야 할 답: corrupted text를 따라간 wrong answer

그래서 DPO는 이 문제에 대해 직접적인 학습 신호를 제공합니다. 정답만 알려주는 것이 아니라, 모델이 실제로 따라간 text shortcut을 명시적으로 밀어내기 때문입니다.

## 5. 메인 결과: Qwen3-VL-8B

메인 실험은 Qwen3-VL-8B-Instruct, DocVQA corrupted held-out split에서 수행했습니다.

| Model | Corrupted Acc | Incorrect aux-hit | Corrected / Regressed |
|---|---:|---:|---:|
| Base Qwen3-VL-8B | 0.715 | 89.5% | - |
| Text-following span DPO, r8 beta 0.1 | 0.900 | 55.0% | 37 / 0 |
| Text-following span DPO, r16 beta 0.1 | 0.915 | 47.1% | 40 / 0 |
| Multi-condition error-mined DPO | 0.920 | 43.8% | 41 / 0 |

여기서 accuracy도 중요하지만, 더 중요한 지표는 `incorrect aux-hit`입니다.

`Incorrect aux-hit`는 모델이 틀렸을 때, 그 틀린 답이 corrupted auxiliary text 안에 있었는지를 보는 값입니다. 이 값이 높으면 모델이 틀릴 때 corrupted text를 따라가는 경우가 많다는 뜻입니다.

Base Qwen3는 틀린 답의 89.5%가 corrupted text 안에 있었습니다. DPO 이후에는 이 값이 47.1%까지 내려갔고, multi-condition error-mined DPO에서는 43.8%까지 내려갔습니다.

따라서 핵심 해석은 다음입니다.

> DPO는 단순히 정답률을 올린 것이 아니라, 모델이 틀리는 방식을 바꾸었다. 특히 corrupted text를 그대로 따라가는 오류가 크게 줄었다.

## 6. Hyperparameter와 추가 DPO 실험

LoRA rank와 beta를 바꾼 결과, beta보다 LoRA capacity가 더 중요했습니다.

| Beta | LoRA rank | Accuracy | Incorrect aux-hit |
|---:|---:|---:|---:|
| 0.1 | 8 | 0.900 | 55.0% |
| 0.2 | 8 | 0.900 | 55.0% |
| 0.1 | 16 | 0.915 | 47.1% |
| 0.2 | 16 | 0.915 | 47.1% |

Prompt-augmented counterfactual DPO는 plain r16 DPO와 같은 0.915를 보였습니다. 반면 각 조건에서 실제로 발생한 base model의 오류를 rejected로 쓰는 multi-condition error-mined DPO는 0.920까지 소폭 개선되었습니다.

이 결과는 중요한 교훈을 줍니다.

> 단순히 학습 데이터를 많이 늘리는 것보다, 모델이 실제로 저지른 오류를 rejected answer로 잡는 것이 더 중요하다.

## 7. SFT와 비교해서 어떻게 이해할 수 있는가

SFT가 나쁘다는 뜻은 아닙니다. SFT는 정답 형식과 task behavior를 맞추는 데 유용할 수 있습니다. 다만 우리가 다루는 핵심 오류는 "정답을 모른다"보다 "충돌 상황에서 잘못된 근거를 믿는다"에 가깝습니다.

그래서 text bias 완화만 놓고 보면 DPO가 더 직접적입니다.

| 방법 | 모델에게 주는 신호 | 이 문제와의 관계 |
|---|---|---|
| SFT | 정답을 출력하라 | 정답 생성은 가르치지만, 어떤 오답을 피해야 하는지는 약함 |
| DPO | 정답이 text-following 오답보다 낫다 | corrupted text shortcut을 직접 밀어냄 |

VQAv2 transfer에서도 pure DPO가 DPO+SFT보다 약간 나았습니다.

| VQAv2 Variant | Corrupted Acc | Incorrect aux-hit | Match | Irrelevant |
|---|---:|---:|---:|---:|
| Base Qwen3-VL-8B | 0.545 | 65.9% | 0.92 | 0.77 |
| Span DPO | 0.645 | 40.8% | 0.90 | 0.75 |
| Span DPO + preservation SFT | 0.635 | 42.5% | 0.90 | 0.75 |

해석은 조심해야 합니다. 이것은 SFT가 항상 불리하다는 뜻이 아닙니다. 다만 이번 문제에서는 이미 명확한 rejected answer가 존재했기 때문에, 정답만 가르치는 SFT보다 정답과 오답을 직접 비교하는 DPO가 더 잘 맞았습니다.

## 8. GRPO가 DPO보다 약했던 이유

GRPO도 RL 계열 방법이지만, 이번 setting에서는 DPO보다 잘 맞지 않았습니다.

DPO는 이미 준비된 두 답을 비교합니다.

> 이미지 기반 정답이 corrupted text를 따라간 오답보다 낫다.

반면 GRPO는 모델이 여러 답을 생성하고, reward로 답들을 평가해야 합니다. 이 방식은 reward 설계와 sampling 품질에 크게 의존합니다.

이번 문제에서는 이미 깨끗한 비교쌍이 있었습니다. 모델이 실제로 낸 text-following wrong answer와 GT answer가 바로 chosen/rejected pair가 됩니다. 이런 경우에는 여러 답을 새로 생성하고 reward를 설계하는 GRPO보다 DPO가 더 직접적이고 노이즈가 적습니다.

DocVQA seed 0 결과도 이 해석과 맞았습니다.

| Variant | Init | Steps | Accuracy | Incorrect aux-hit | Corrected / Regressed |
|---|---|---:|---:|---:|---:|
| Base Qwen3-VL-8B | - | - | 0.715 | 89.5% | - |
| GRPO-only, short pilot | base | 100 | 0.710 | 89.7% | 0 / 1 |
| GRPO-only, longer training | base | 300 | 0.705 | 89.8% | 0 / 2 |
| GRPO-only, dense reward | base | 300 | 0.705 | 89.8% | 0 / 2 |
| Text-following span DPO | base | 700 | 0.900 | 55.0% | 37 / 0 |
| Span DPO -> GRPO | DPO | 150 | 0.900 | 60.0% | 37 / 0 |

GRPO가 원리적으로 불가능하다는 뜻은 아닙니다. 다만 이 문제에서는 high-precision pair를 자동으로 만들 수 있었고, 그 결과 DPO가 더 안정적인 선택이었습니다.

## 9. 다른 모델에서도 통하는가

Qwen3에서 얻은 결과가 다른 VLM에도 이어지는지 확인하기 위해 Qwen2-VL-7B, LLaVA-Next-7B, LLaVA-Next-13B를 추가로 평가했습니다. 모두 RTX 6000 Ada에서 실행했고 H100은 사용하지 않았습니다.

| Model | Setting | Corrupted Acc | Incorrect aux-hit | Corrected / Regressed | Match | Irrelevant |
|---|---|---:|---:|---:|---:|---:|
| Qwen2-VL-7B | base | 0.570 | 93.0% | - | - | - |
| Qwen2-VL-7B | span DPO | 0.775 | 86.7% | 41 / 0 | 0.98 | 0.94 |
| LLaVA-Next-7B | base | 0.105 | 90.5% | - | - | - |
| LLaVA-Next-7B | span DPO | 0.100 | 91.7% | 1 / 2 | 0.95 | 0.59 |
| LLaVA-Next-13B | base | 0.115 | 92.7% | - | - | - |
| LLaVA-Next-13B | span DPO | 0.130 | 95.4% | 3 / 0 | 0.97 | 0.62 |

Qwen2-VL-7B에서는 강한 positive transfer가 나타났습니다. Accuracy가 0.570에서 0.775로 올랐고, 41개를 교정하는 동안 regression은 없었습니다. 이는 Qwen3에서 본 효과가 Qwen3에만 특화된 현상은 아니라는 점을 보여줍니다.

반면 LLaVA-Next 계열에서는 결과가 좋지 않았습니다. 7B는 DPO 이후 accuracy가 0.105에서 0.100으로 내려갔고, 13B는 0.115에서 0.130으로 아주 조금 올랐지만 incorrect aux-hit가 92.7%에서 95.4%로 오히려 증가했습니다. 즉 13B는 몇 개의 정답을 추가로 맞혔지만, 남은 오답들은 더 강하게 corrupted text 안에 머물렀습니다.

이 결과는 중요한 경계 조건을 보여줍니다.

> Text-following span DPO는 VLM에게 DocVQA 능력을 처음부터 가르치는 방법이 아니다. Base model이 이미 이미지를 읽고 답할 수 있을 때, corrupted text에 끌려가는 conflict-resolution 오류를 교정하는 방법이다.

Qwen 계열은 base 성능이 어느 정도 있었기 때문에 DPO가 잘 작동했습니다. 반면 LLaVA-Next는 현재 prompt/template에서 DocVQA base accuracy가 0.10 수준으로 매우 낮았습니다. 이런 상태에서는 DPO가 text bias를 줄이기보다, 낮은 task alignment 위에서 일부 답만 바꾸는 데 그쳤습니다.

## 10. 발표용 핵심 메시지

PPT에서는 다음 흐름으로 설명하는 것이 가장 설득력 있습니다.

1. 실제 VLM 응용에서는 이미지와 함께 OCR/retrieved/caption text가 들어가고, 이 텍스트는 틀릴 수 있다.
2. 텍스트가 틀렸을 때 VLM이 이미지를 확인하지 않고 텍스트를 따라가면, 이는 단순 성능 문제가 아니라 multimodal grounding 실패다.
3. 우리는 이 현상을 text-following error로 자동 탐지했다. 모델의 오답이 corrupted text 안에 있으면, 사람이 라벨링하지 않아도 high-precision failure로 볼 수 있다.
4. 이 failure를 DPO pair로 바꾸면, 이미지 기반 정답을 chosen으로, corrupted text 기반 오답을 rejected로 둘 수 있다.
5. 그래서 DPO는 naive한 RL 선택이 아니라, 이미지와 텍스트가 충돌할 때 무엇을 믿을지 학습시키는 자연스러운 방법이다.
6. Qwen3/Qwen2에서는 accuracy가 크게 오르고 text-following error가 줄었다.
7. GRPO는 reward와 sampling에 의존하기 때문에, 이미 깨끗한 chosen/rejected pair가 있는 이 문제에서는 DPO보다 불안정했다.
8. LLaVA 결과는 이 방법의 한계도 보여준다. Base VLM이 task를 거의 풀지 못하면 DPO만으로는 충분하지 않다.

## 11. 전체 결론

이번 실험의 가장 중요한 결론은 다음입니다.

> VLM의 text bias는 단순한 노이즈 강건성 문제가 아니라, 이미지와 텍스트가 충돌할 때 텍스트를 과하게 믿는 선택 오류다.

우리는 이 선택 오류를 `text-following error`로 구체화했고, 이를 자동으로 DPO pair로 바꾸었습니다. 이 방식은 manual labeling 없이도 모델이 실제로 저지른 오류를 학습 신호로 사용할 수 있다는 장점이 있습니다.

Qwen3-VL-8B에서는 base accuracy 0.715가 DPO 후 0.900~0.920까지 올랐고, incorrect aux-hit는 89.5%에서 43.8~55.0% 수준으로 크게 내려갔습니다. Qwen2-VL-7B에서도 0.570에서 0.775로 개선되어 같은 방향의 효과가 반복되었습니다.

반면 GRPO는 개선을 만들지 못했고, DPO 이후 추가 GRPO도 이득을 주지 못했습니다. 이는 이번 문제에서는 reward를 설계해 여러 sampled answer를 비교하는 것보다, 실제 text-following wrong answer와 GT answer를 직접 비교하는 DPO가 더 적합하다는 점을 보여줍니다.

LLaVA-Next 결과는 방법의 한계를 보여줍니다. Base DocVQA 성능이 매우 낮은 모델에서는 DPO만으로 text bias를 줄이기 어렵습니다. 따라서 이 방법은 task ability를 새로 만드는 방법이 아니라, 이미 어느 정도 task를 수행할 수 있는 VLM에서 text shortcut을 줄이는 방법으로 이해하는 것이 가장 정확합니다.

최종적으로 이 연구의 주장은 다음처럼 정리할 수 있습니다.

> Text bias를 줄이려면 모든 오류를 무작정 학습시키는 것보다, corrupted text를 실제로 따라간 high-precision failure를 골라내고, 이를 image-grounded answer와 직접 비교시키는 preference learning이 효과적이다. 이 문제에서는 SFT보다 DPO가 더 직접적인 신호를 제공했고, GRPO보다 안정적으로 text-following behavior를 줄였다. 다만 이 효과는 base VLM이 이미 시각 정보와 task를 어느 정도 이해할 때 가장 잘 나타난다.
