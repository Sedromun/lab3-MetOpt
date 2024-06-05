import math
from enum import Enum
import numpy as np


class LearningRateScheduling(Enum):
    CONSTANT = 0,
    EXPONENTIAL = 1,
    STEPPED = 2


class LinearRegression:
    def __init__(self, X, y, epochs: int, learning_rate: float, batch_size: int,
                 scheduling: LearningRateScheduling = LearningRateScheduling.CONSTANT,
                 epoch_drop: int = 10, drop: float = 0.5, exp_k: float = 0.1):
        self.X = X
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.batch_size = batch_size
        self.scheduling = scheduling
        self.w = [1 for _ in range(X.shape[1] + 1)]
        self.ws = [self.w]
        self.errors = []
        self.epoch_drop = epoch_drop
        self.drop = drop
        self.exp_k = exp_k

    def update_learning_rate(self, epoch):
        if self.scheduling == LearningRateScheduling.STEPPED:
            if epoch % self.epoch_drop == 0:
                self.learning_rate = self.learning_rate * self.drop
        elif self.scheduling == LearningRateScheduling.EXPONENTIAL:
            self.learning_rate = self.learning_rate * np.exp(-epoch * self.exp_k)

    def fit(self):
        self.X = np.hstack((self.X, np.array([[1] for _ in range(self.X.shape[0])])))
        for epoch in range(self.epochs):
            self.__shuffle()
            self.__SGD(self.learning_rate)
            self.update_learning_rate(epoch)

    def __shuffle(self):
        permutation = [i for i in range(self.X.shape[0])]
        np.random.shuffle(permutation)
        new_X = []
        new_Y = []
        for i in range(self.X.shape[0]):
            new_X.append(self.X[permutation[i]])
            new_Y.append(self.y[permutation[i]])
        self.X = np.array(new_X)
        self.y = np.array(new_Y)

    def __MSE(self, deviation: [float]) -> float:
        return np.sum(deviation ** 2)

    def __SGD(self, learning_rate: float):
        batches_X = [self.X[i:i + self.batch_size] for i in range(0, len(self.X), self.batch_size)]
        batches_y = [self.y[i:i + self.batch_size] for i in range(0, len(self.y), self.batch_size)]
        for i in range(len(batches_X)):
            batch_x = batches_X[i]
            batch_y = batches_y[i]
            predictions = np.dot(batch_x, self.w)
            dev = predictions - batch_y
            gradient = 2 * batch_x.T.dot(dev) / self.X.shape[0]
            self.w -= learning_rate * gradient

            self.ws.append(self.w)
            self.errors.append(self.__MSE(dev))

    def get_error(self):
        predictions = np.dot(self.X, self.w)
        dev = predictions - self.y
        return self.__MSE(dev) / self.X.shape[0]

    def predict(self, x: [float]) -> [float]:
        return np.dot(np.transpose(x), self.get_weights()) + self.get_bias()

    def get_weights(self):
        return self.w[:-1]

    def get_bias(self):
        return self.w[-1]

    def get_errors(self):
        return self.errors

    def get_weights_history(self):
        return self.ws
