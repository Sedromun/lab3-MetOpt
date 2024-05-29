from sklearn.datasets import make_regression

from generation.degenerator import load_static_file
from methods_module.lib_regression_methods import create, predict
from methods_module.linear_regression import LinearRegression, LearningRateScheduling
from visualisation_module.visualisation import *


if __name__ == '__main__':
    data = load_static_file("noised_data.json")

    # X = np.array([[1], [1.5], [2], [2], [3], [3.3], [4], [5]])
    # y = np.array([0.9, 1.4, 2.1, 2, 3.1, 3.25, 4.2, 4.9])

    X = np.array(data['points'])
    y = np.array(data['values'])

    lr = LinearRegression(X, y, epochs=100, learning_rate=0.05, batch_size=20,
                          scheduling=LearningRateScheduling.STEPPED)
    lr.fit()
    print(lr.get_weights())
    print(lr.get_bias())
    print(lr.predict(X[0]))
