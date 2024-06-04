from enum import Enum
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np


class LearningRateScheduling(Enum):
    CONSTANT = 0,
    EXPONENTIAL = 1,
    STEPPED = 2


class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, epochs=1000, batch_size=32, regularization=None, lambda_=0.01, alpha=0.5):
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_ = lambda_
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        poly = PolynomialFeatures(degree=self.degree)
        X_poly = poly.fit_transform(X)
        n_samples, n_features = X_poly.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_poly[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_pred = np.dot(X_batch, self.weights) + self.bias

                dw = (2 / X_batch.shape[0]) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (2 / X_batch.shape[0]) * np.sum(y_pred - y_batch)

                if self.regularization == 'l2':
                    dw += (2 * self.lambda_) * self.weights
                elif self.regularization == 'l1':
                    dw += self.lambda_ * np.sign(self.weights)
                elif self.regularization == 'elasticnet':
                    dw += (self.alpha * self.lambda_ * np.sign(self.weights)) + (
                                (1 - self.alpha) * self.lambda_ * self.weights)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        poly = PolynomialFeatures(degree=self.degree)
        X_poly = poly.fit_transform(X)
        return np.dot(X_poly, self.weights) + self.bias

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_errors(self):
        return self.errors

    def get_weights_history(self):
        return self.ws
