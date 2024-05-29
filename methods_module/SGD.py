import numpy as np


class LinearRegression:
    def __init__(self, X, y, epochs: int, learning_rate: float, batch_size: int):
        self.X = X
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.w = [0 for _ in range(X.shape[1] + 1)]
        self.ws = [self.w]
        self.errors = []

    def fit(self):
        self.X = np.hstack((self.X, np.array([[1] for _ in range(self.X.shape[0])])))
        for epoch in range(self.epochs):
            self.__shuffle()
            self.__SGD(self.learning_rate)

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

    def __MSE(self, y: [float], x: [float]) -> float:
        return np.sum((y - x) ** 2)

    def __SGD(self, learning_rate: float):
        batches_X = [self.X[i:i + self.batch_size] for i in range(0, len(self.X), self.batch_size)]
        batches_y = [self.y[i:i + self.batch_size] for i in range(0, len(self.y), self.batch_size)]
        for i in range(len(batches_X)):
            batch_x = batches_X[i]
            batch_y = batches_y[i]
            predictions = np.dot(batch_x, self.w)
            gradient = 2 * batch_x.T.dot(predictions - batch_y) / batch_x.shape[0]

            self.w -= learning_rate * gradient
            self.ws.append(self.w)
            self.errors.append(self.__MSE(batch_y, batch_x))

    def predict(self, x: [float]) -> [float]:
        return np.dot(x, self.get_weights()) + self.get_bias()

    def get_weights(self):
        return self.w[:-1]

    def get_bias(self):
        return self.w[-1]

    def get_errors(self):
        return self.errors

    def get_weights_history(self):
        return self.ws