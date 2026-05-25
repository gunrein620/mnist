# -*- coding: utf-8 -*-
"""학습 루프, 평가, 시각화 함수 모음."""

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """
    미니배치 학습 루프.

    한 배치마다 Forward -> Loss -> Backward -> Optimizer 업데이트 순서로 진행합니다.
    교육생은 이 함수에서 "예측값을 만들고, 손실을 계산하고, gradient로 파라미터를 바꾸는"
    전체 흐름을 확인할 수 있습니다.

    Returns:
        loss_history: epoch별 평균 손실 리스트
    """
    # TODO: epoch마다 데이터를 섞고, batch 단위로 forward/loss/backward/update를 수행하세요.
    # 힌트: Softmax + CrossEntropy 결합 gradient는 y_pred copy에서 정답 위치에 1을 빼서 만듭니다.
    loss_history = []  # epoch마다 평균 loss를 기록합니다.
    num_train = x_train.shape[0]  # 전체 학습 샘플 수입니다.

    for _ in range(epochs):
        indices = np.random.permutation(num_train)  # 매 epoch마다 데이터 순서를 섞습니다.
        epoch_loss = 0.0  # batch loss를 샘플 수 기준으로 누적합니다.

        for start in range(0, num_train, batch_size):
            batch_indices = indices[start : start + batch_size]  # 이번 batch에 사용할 인덱스입니다.
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]
            batch_len = x_batch.shape[0]  # 마지막 batch는 batch_size보다 작을 수 있습니다.

            y_pred = model.forward(x_batch, train=True)  # 순전파로 클래스 확률을 구합니다.
            loss = cross_entropy_loss(y_pred, y_batch)  # 현재 batch의 평균 손실입니다.

            dout = y_pred.copy()  # Softmax + CrossEntropy의 출력층 gradient를 만듭니다.
            dout[np.arange(batch_len), y_batch] -= 1  # 정답 클래스 위치만 1을 뺍니다.
            dout /= batch_len  # batch 평균 gradient로 맞춥니다.

            model.backward(dout)  # 역전파로 각 파라미터의 gradient를 계산합니다.
            optimizer.update(model.params, model.grads)  # gradient 방향의 반대로 파라미터를 갱신합니다.

            epoch_loss += loss * batch_len  # epoch 평균을 위해 샘플 수만큼 가중해 더합니다.

        loss_history.append(epoch_loss / num_train)  # epoch 하나의 평균 loss입니다.

    return loss_history


def evaluate(model, x, y):
    """정확도(%)와 총 파라미터 수 반환."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """손실 커브 그래프."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
