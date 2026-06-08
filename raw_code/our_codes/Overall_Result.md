# Overall Result: Text Bias를 줄이기 위한 Preference Learning 정리

## 1. 문제를 어떻게 보았는가

이번 실험의 출발점은 단순합니다. Vision-Language Model이 이미지와 텍스트를 함께 볼 때, 텍스트가 틀렸더라도 그럴듯해 보이면 텍스트를 너무 쉽게 믿는다는 것입니다.

예를 들어 문서 이미지 안에는 실제 정답이 `DWRITE 077665`로 적혀 있는데, 함께 제공된 corrupted text에는 `DOCNUM 123456` 같은 잘못된 답이 들어있을 수 있습니다. 이때 모델이 이미지를 보고 정답을 읽는 대신, corrupted text에 있는 그럴듯한 답을 그대로 따라가면 이것을 text bias라고 볼 수 있습니다.

따라서 우리는 text bias를 막연하게 "텍스트 때문에 성능이 떨어졌다"라고 보지 않았습니다. 더 구체적으로는 다음과 같이 정의했습니다.

> 모델이 틀린 답을 냈고, 그 틀린 답이 corrupted auxiliary text 안에 실제로 등장한다면, 이것은 텍스트를 과신해서 생긴 오류일 가능성이 높다.

이 정의가 중요한 이유는 사람이 직접 "이 샘플은 text bias다"라고 라벨링하지 않아도 된다는 점입니다. 모델의 예측값과 corrupted text를 비교하면, text-following error를 자동으로 골라낼 수 있습니다.

## 2. 우리가 채택한 방법

우리가 메인으로 채택한 방법은 text-following span DPO입니다. 이름은 조금 길지만 아이디어는 어렵지 않습니다.

1. 먼저 기본 VLM을 corrupted text가 포함된 조건에서 평가합니다.
2. 모델이 틀린 경우만 모읍니다.
3. 그중에서도 모델의 오답이 corrupted text 안에 실제로 등장하는 경우만 고릅니다.
4. 이 샘플을 preference pair로 바꿉니다.

여기서 preference pair는 "둘 중 무엇을 더 선호해야 하는가"를 알려주는 학습 데이터입니다.

| 항목 | 의미 |
|---|---|
| chosen | 이미지에 근거한 정답 |
| rejected | corrupted text를 따라간 오답 |

즉 모델에게 이런 메시지를 주는 것입니다.

> 이 상황에서는 corrupted text에 있는 그럴듯한 답보다, 이미지에서 읽을 수 있는 정답을 더 믿어야 한다.

이 방식은 정답만 알려주는 것이 아니라, "어떤 답을 선택하면 안 되는지"까지 같이 알려줍니다. 이 점이 핵심입니다.

## 3. 왜 이 방식이 통하는가

VLM이 text bias를 보이는 경우를 보면, 모델이 아무것도 몰라서 틀리는 것이 아닙니다. 오히려 이미지와 텍스트 사이에 충돌이 있을 때, 어느 쪽을 더 믿어야 할지 잘못 판단하는 경우가 많습니다.

일반적인 지도학습은 보통 "이 질문의 정답은 이것이다"라고 알려줍니다. 하지만 우리의 문제는 조금 다릅니다.

우리가 해결하고 싶은 것은 다음 질문입니다.

> 이미지에는 A가 있고, 텍스트에는 B가 있을 때, 모델은 왜 B를 따라가는가? 그리고 어떻게 A를 선택하도록 만들 수 있는가?

DPO는 이 문제와 잘 맞습니다. 왜냐하면 DPO는 단순히 정답을 외우게 하는 것이 아니라, 두 답 중 어느 쪽이 더 나은지 비교하게 만들기 때문입니다.

우리 데이터에서는 비교 대상이 매우 분명합니다.

- 더 좋은 답: 이미지에 근거한 GT answer
- 피해야 할 답: corrupted text에서 따라온 wrong answer

그래서 DPO가 효과적이었습니다. 모델은 "정답을 출력하라"만 배우는 것이 아니라, "corrupted text 안에 있는 그럴듯한 오답을 따라가지 말라"는 방향까지 같이 배우게 됩니다.

## 4. 주요 결과

메인 실험은 Qwen3-VL-8B-Instruct를 사용했고, DocVQA corrupted held-out split에서 평가했습니다.

| Model | Corrupted Acc | Incorrect aux-hit | Corrected / Regressed |
|---|---:|---:|---:|
| Base Qwen3-VL-8B | 0.715 | 89.5% | - |
| Text-following span DPO, r8 beta 0.1 | 0.900 | 55.0% | 37 / 0 |
| Text-following span DPO, r16 beta 0.1 | 0.915 | 47.1% | 40 / 0 |
| Multi-condition error-mined DPO | 0.920 | 43.8% | 41 / 0 |

여기서 accuracy도 중요하지만, 더 중요한 지표는 incorrect aux-hit입니다.

Incorrect aux-hit는 모델이 틀렸을 때, 그 틀린 답이 corrupted text 안에 있었는지를 보는 값입니다. 이 값이 높으면, 모델이 틀릴 때 corrupted text를 그대로 따라가는 경우가 많다는 뜻입니다.

Base model은 틀린 답의 89.5%가 corrupted text 안에 있었습니다. 즉 틀릴 때 대부분 텍스트를 따라간 것입니다. DPO 이후에는 이 값이 47.1%, multi-condition DPO에서는 43.8%까지 내려갔습니다.

따라서 결과를 이렇게 해석할 수 있습니다.

> DPO는 단순히 정답률을 올린 것이 아니라, 모델이 틀리는 방식을 바꾸었다. 특히 corrupted text를 그대로 따라가는 오류가 크게 줄었다.

## 5. SFT보다 RL 계열 방법이 더 적합한 이유

여기서 말하는 RL은 복잡한 강화학습 전체라기보다, "어떤 답을 더 선호해야 하는지"를 학습시키는 preference learning에 가깝습니다.

SFT는 보통 다음과 같은 방식입니다.

> 이 입력이 들어오면 이 정답을 출력하라.

반면 DPO는 다음에 가깝습니다.

> 이 답은 더 좋고, 저 답은 그럴듯해 보여도 선택하면 안 된다.

우리 문제에서는 이 차이가 중요합니다. Text bias는 모델이 답을 아예 몰라서 생기는 문제라기보다, 이미지와 텍스트가 충돌할 때 잘못된 쪽을 선택해서 생기는 문제입니다.

따라서 정답만 알려주는 SFT보다, 정답과 피해야 할 오답을 함께 비교시키는 DPO가 더 자연스럽습니다.

실험적으로도 SFT를 preservation anchor로 섞은 방식은 큰 이득을 주지 못했습니다. VQAv2에서 DPO+SFT는 pure DPO보다 오히려 약간 낮은 결과를 보였습니다.

| VQAv2 Variant | Corrupted Acc | Incorrect aux-hit | Match | Irrelevant |
|---|---:|---:|---:|---:|
| Base Qwen3-VL-8B | 0.545 | 65.9% | 0.92 | 0.77 |
| Span DPO | 0.645 | 40.8% | 0.90 | 0.75 |
| Span DPO + preservation SFT | 0.635 | 42.5% | 0.90 | 0.75 |

이 결과는 SFT가 항상 나쁘다는 뜻은 아닙니다. 다만 우리가 다루는 text bias 문제에서는, 단순히 정답 생성을 더 학습시키는 것보다 "정답과 text-following 오답을 비교시키는 것"이 더 직접적인 신호였다고 볼 수 있습니다.

## 6. DPO가 GRPO보다 잘 맞았던 이유

GRPO도 RL 계열 방법이지만, 이번 문제에서는 DPO보다 잘 맞지 않았습니다.

쉽게 말하면 DPO는 이미 준비된 두 답을 비교합니다.

> 정답이 이 오답보다 낫다.

반면 GRPO는 모델이 여러 답을 직접 생성하게 한 뒤, reward를 계산해서 어느 답이 나은지 알려주는 방식입니다. 이 방식은 reward 설계가 매우 중요합니다.

이번 문제에서는 이미 좋은 비교쌍이 있었습니다.

- chosen: 이미지 기반 정답
- rejected: corrupted text를 따라간 오답

즉 굳이 여러 답을 새로 생성하고 reward로 점수를 매기지 않아도, 학습해야 할 비교 관계가 아주 명확했습니다. 이런 상황에서는 DPO가 더 직접적이고 노이즈가 적습니다.

GRPO에서는 모델이 생성한 답들이 충분히 다양하거나 좋은 후보를 포함해야 합니다. 그런데 실제 학습 중에는 correct sampled answer가 자주 나오지 않았고, reward를 더 촘촘하게 만들어도 held-out greedy prediction은 거의 움직이지 않았습니다.

## 7. GRPO 실험 결과

DocVQA seed 0 disjoint split 기준 결과는 다음과 같습니다.

| Variant | Init | Steps | Accuracy | Incorrect aux-hit | Corrected / Regressed |
|---|---|---:|---:|---:|---:|
| Base Qwen3-VL-8B | - | - | 0.715 | 89.5% | - |
| GRPO-only, short pilot | base | 100 | 0.710 | 89.7% | 0 / 1 |
| GRPO-only, longer training | base | 300 | 0.705 | 89.8% | 0 / 2 |
| GRPO-only, dense reward | base | 300 | 0.705 | 89.8% | 0 / 2 |
| Text-following span DPO | base | 700 | 0.900 | 55.0% | 37 / 0 |
| Span DPO -> GRPO, longer-training reward | DPO | 150 | 0.900 | 60.0% | 37 / 0 |

각 GRPO 설정의 차이는 다음과 같습니다.

| 이름 | 무엇을 바꿨는가 | 의도 |
|---|---|---|
| GRPO-only, short pilot | base model에서 100 step만 짧게 학습 | GRPO가 기본적으로 text-following behavior를 움직일 수 있는지 빠르게 확인 |
| GRPO-only, longer training | base model에서 300 step으로 학습 길이와 learning rate를 늘림 | 짧아서 효과가 없었던 것인지 확인 |
| GRPO-only, dense reward | 300 step은 유지하되, 정답 보상과 corrupted text copy penalty를 더 촘촘하게 설계 | reward가 너무 sparse해서 학습이 안 된 것인지 확인 |
| Span DPO -> GRPO | 먼저 DPO로 좋아진 모델에서 GRPO를 추가 학습 | DPO가 만든 좋은 출발점 위에서 GRPO가 추가 개선을 줄 수 있는지 확인 |

이 결과를 보면 GRPO-only는 base model의 text-following behavior를 거의 바꾸지 못했습니다. Dense reward를 사용해도 결과가 크게 달라지지 않았습니다.

DPO 이후에 GRPO를 추가한 경우도 accuracy를 더 올리지 못했습니다. 오히려 incorrect aux-hit는 DPO-only보다 나빠졌습니다.

따라서 현재 실험에서는 다음과 같이 정리하는 것이 가장 적절합니다.

> GRPO가 원리적으로 불가능한 방법이라는 뜻은 아니다. 다만 이 문제에서는 이미 깨끗한 chosen/rejected pair를 만들 수 있었기 때문에, reward를 설계해서 여러 generated answer를 비교하는 GRPO보다 DPO가 훨씬 직접적이었다.

## 8. 추가 ablation에서 얻은 교훈

### LoRA rank와 beta

LoRA rank를 8에서 16으로 늘리면 성능이 조금 더 좋아졌습니다.

| Beta | LoRA rank | Accuracy | Incorrect aux-hit |
|---:|---:|---:|---:|
| 0.1 | 8 | 0.900 | 55.0% |
| 0.2 | 8 | 0.900 | 55.0% |
| 0.1 | 16 | 0.915 | 47.1% |
| 0.2 | 16 | 0.915 | 47.1% |

반면 beta를 0.1에서 0.2로 늘리는 것은 큰 차이를 만들지 않았습니다. 현재까지는 beta 조정보다 LoRA capacity를 조금 늘리는 쪽이 더 효과적이었습니다.

### Counterfactual prompt augmentation

처음에는 corrupted, no-text, match, irrelevant 조건으로 prompt를 늘려보았습니다. 하지만 rejected answer는 여전히 corrupted text에서 나온 같은 오답을 사용했습니다.

결과는 plain r16 DPO와 같았습니다.

| Variant | Accuracy | Incorrect aux-hit |
|---|---:|---:|
| Plain span DPO r16 | 0.915 | 47.1% |
| Prompt-augmented counterfactual DPO | 0.915 | 47.1% |

이 결과는 단순히 prompt 조건만 늘리는 것은 충분하지 않다는 것을 보여줍니다. 중요한 것은 각 조건에서 실제로 모델이 어떤 잘못을 했는지에 맞는 rejected answer를 구성하는 것입니다.

### Multi-condition error-mined DPO

그래서 다음으로는 각 조건에서 base model이 실제로 틀린 답을 rejected로 사용했습니다.

이 방식은 plain r16 DPO보다 아주 작게 좋아졌습니다.

| Variant | Accuracy | Incorrect aux-hit |
|---|---:|---:|
| Plain span DPO r16 | 0.915 | 47.1% |
| Multi-condition error-mined DPO | 0.920 | 43.8% |

개선폭은 작지만 방향은 좋았습니다. 다만 DocVQA에서는 match/no-text 조건의 실패 수가 적어서 추가 신호가 제한적이었습니다. 따라서 이 결과는 main method라기보다 "DPO pair의 질이 중요하다"는 것을 보여주는 보조 실험으로 보는 것이 적절합니다.

## 9. 전체 결론

이번 실험에서 가장 중요한 결론은 다음입니다.

> VLM의 text bias는 단순한 성능 저하가 아니라, 이미지와 텍스트가 충돌할 때 텍스트를 더 믿어버리는 선택 오류로 볼 수 있다.

이 관점에서 보면 DPO가 왜 효과적인지 자연스럽게 설명됩니다. DPO는 모델에게 정답만 알려주는 것이 아니라, corrupted text에서 온 그럴듯한 오답보다 이미지 기반 정답을 더 선호하도록 직접 학습시킵니다.

실험적으로도 DocVQA에서 base model은 corrupted condition에서 0.715 accuracy를 보였지만, text-following span DPO는 0.900까지 올랐고, LoRA rank 16에서는 0.915까지 개선되었습니다. Multi-condition error-mined DPO는 0.920까지 소폭 추가 개선되었습니다.

더 중요한 것은 text-following error의 감소입니다. Base model은 틀릴 때 대부분 corrupted text 안의 답을 따라갔지만, DPO 이후 이 비율이 크게 줄었습니다. 즉 DPO는 단순히 더 많이 맞힌 것이 아니라, 모델이 corrupted text에 끌려가는 오류 자체를 줄였습니다.

SFT는 정답을 가르치는 데는 자연스럽지만, "그럴듯한 오답을 피하라"는 신호가 약합니다. GRPO는 reward 설계를 통해 이 문제를 다룰 수는 있지만, 이번 setting에서는 이미 명확한 chosen/rejected pair가 있었기 때문에 DPO보다 간접적이고 불안정했습니다.

따라서 현재 가장 설득력 있는 스토리는 다음과 같습니다.

> Text bias를 줄이기 위해서는 모든 오류를 무작정 학습시키는 것보다, corrupted text를 실제로 따라간 high-precision failure를 골라내고, 이를 image-grounded answer와 직접 비교시키는 preference learning이 효과적이다. 이 문제에서는 SFT보다 DPO가 더 직접적인 학습 신호를 제공하며, GRPO보다도 더 안정적으로 text-following behavior를 줄였다.
