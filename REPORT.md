# MNIST 손글씨 인식 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| **반** | SW-AI 301반 |
| **팀원** | 고민석, 나지운, 박건우, 박민석 |

---

## 1. 실험 목적

본 과제의 목적은 PyTorch, TensorFlow 같은 딥러닝 프레임워크 없이 NumPy만으로 MNIST 손글씨 숫자 분류기를 구현하는 것이다.

이를 위해 신경망의 핵심 구성 요소인 Affine 계층, ReLU, Softmax, Cross Entropy Loss, SGD, Adam, BatchNorm, Dropout, 학습 루프를 직접 구현했다. 단순히 정확도를 얻는 것뿐 아니라 `forward -> loss -> backward -> update` 흐름에서 각 계층이 어떤 값을 계산하고 저장하는지 이해하는 데 초점을 두었다.

---

## 2. 모델 구조

| 구분 | 내용 |
| --- | --- |
| **입력** | 784차원 벡터 (28x28 픽셀 이미지를 flatten, 0~1 정규화) |
| **은닉층 1** | Affine(784 -> 512) -> BatchNorm -> ReLU -> Dropout |
| **은닉층 2** | Affine(512 -> 256) -> BatchNorm -> ReLU -> Dropout |
| **출력층** | Affine(256 -> 10) -> Softmax |

전체 구조는 다음과 같다.

```text
Input(784)
-> Affine(512)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(256)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(10)
-> Softmax
```

은닉층 활성화 함수로 ReLU를 사용했기 때문에 가중치는 He initialization으로 초기화했다.

---

## 3. 학습 설정

| 항목 | 값 |
| --- | --- |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Epochs** | 20 |
| **Batch Size** | 128 |
| **Dropout 비율** | 0.5 |
| **BatchNorm Momentum** | 0.9 |
| **가중치 초기화** | He initialization |
| **손실 함수** | Cross Entropy Loss |

Epoch는 전체 학습 데이터를 한 번 모두 보는 단위이며, 20 epoch는 전체 데이터를 20번 반복해서 학습했다는 의미이다.

각 mini-batch마다 다음 순서로 학습을 수행했다.

```text
Forward -> Loss -> Backward -> Optimizer Update
```

- **Forward**: 입력 이미지를 모델에 넣어 예측값을 계산한다.
- **Loss**: 예측값이 정답과 얼마나 다른지 Cross Entropy Loss로 계산한다.
- **Backward**: 손실을 바탕으로 각 가중치를 어느 방향으로 고쳐야 할지 gradient를 계산한다.
- **Update**: 계산된 gradient를 이용해 Adam optimizer가 가중치와 편향을 수정한다.

즉, 문제를 풀고, 채점하고, 틀린 이유를 되짚은 다음, 다음에는 더 잘 풀도록 파라미터를 고치는 과정이다.

---

## 4. 실험 환경

| 항목 | 내용 |
| --- | --- |
| **Python** | Python 3.11 |
| **주요 라이브러리** | NumPy, Matplotlib |
| **테스트 도구** | Pytest |
| **실행 환경** | Google Colab T4 GPU |
| **학습 소요 시간** | 약 4분 |
| **단위 테스트 실행 시간** | 3.45초 |


---

## 5. 결과

| 항목 | 값 |
| --- | --- |
| **테스트 정확도** | 98.54% |
| **총 파라미터 수** | 537,354 |

과제의 권장 목표는 97% 이상이었으며, 최종 모델은 테스트 정확도 98.54%로 목표 기준을 넘겼다.

테스트 결과:

```text
21 passed in 3.45s
```

총 파라미터 수는 다음과 같이 계산된다.

```text
W1: 784 x 512 = 401,408
b1: 512
W2: 512 x 256 = 131,072
b2: 256
W3: 256 x 10 = 2,560
b3: 10
BatchNorm gamma/beta: (512 + 512) + (256 + 256) = 1,536
Total = 537,354
```

### 정확도 향상을 위한 설정

이 정확도를 얻기 위해 단순히 층을 크게 만드는 것보다, 학습이 안정적으로 진행되도록 만드는 데 초점을 두었다.

먼저 He initialization은 처음 가중치 값을 정하는 방법이다. 가중치가 처음부터 너무 크거나 작으면 학습이 불안정해질 수 있기 때문에, ReLU에 잘 맞는 He initialization을 적용했다.

BatchNorm은 각 층을 지나는 값들의 분포를 정리해주는 방법이다. 층을 여러 번 지나면서 값의 분포가 흔들리면 학습이 어려워질 수 있기 때문에, BatchNorm을 사용해 학습을 더 안정적으로 만들었다.

Dropout은 학습 중 일부 뉴런을 잠깐 꺼주는 방법이다. 모델이 특정 뉴런 조합에만 의존하면 학습 데이터에는 잘 맞지만 새로운 데이터에는 약해질 수 있기 때문에, Dropout으로 과적합을 줄이고자 했다.

Adam은 가중치를 업데이트하는 방법이다. 파라미터마다 업데이트 정도를 조절해주기 때문에, 더 안정적으로 학습되도록 도와준다.

정리하면 He initialization, BatchNorm, Dropout, Adam을 사용해 가중치 초기화, 학습 안정화, 과적합 방지, 업데이트 안정성을 챙기고자 했다.

### 손실 커브

아래 학습 곡선에서 초반에는 loss가 빠르게 감소하고, 후반으로 갈수록 완만하게 줄어드는 것을 확인했다.

초반에는 모델이 아직 거의 학습되지 않은 상태라 명확하게 틀리는 예측이 많기 때문에, 몇 번의 학습만으로도 loss가 크게 줄어든다. 후반으로 갈수록 쉬운 패턴은 어느 정도 학습된 상태이므로, 서로 비슷하게 생긴 숫자처럼 더 어려운 예시들이 남아 개선 폭이 작아진다.

이 흐름을 통해 모델이 정상적으로 수렴하고 있다고 볼 수 있다.

![Training Loss Curve](assets/loss_curve.png)

---

## 6. 회고

이번 과제에서는 신경망의 주요 구성 요소를 NumPy 배열 연산만으로 직접 구현했다. `Affine.forward`에서는 `x @ W + b`로 선형 변환을 수행했고, `Affine.backward`에서는 `dW`, `db`, `dx`의 shape가 각각 원래 파라미터 및 입력과 일치하는지 확인했다.

ReLU는 forward에서 양수 위치를 mask로 저장하고, backward에서 해당 위치로만 gradient가 흐르게 했다. Softmax는 overflow 방지를 위해 row별 최댓값을 빼고 확률을 계산했다.

BatchNorm은 학습 모드와 추론 모드를 분리했다. 학습 모드에서는 batch mean/variance를 사용하고 running mean/variance를 갱신했으며, 추론 모드에서는 running 통계를 사용했다. Dropout은 학습 모드에서 random mask를 적용하고, 추론 모드에서는 평균 출력 크기에 맞춰 scale했다.

Google Colab T4 GPU 환경에서 약 4분 동안 20 epoch 학습을 완료했고, 98.54%의 테스트 정확도를 확인했다. 이를 통해 구현한 순전파, 역전파, Adam 업데이트, BatchNorm, Dropout이 전체 학습 과정에서 정상적으로 연결됨을 확인할 수 있었다.
