from generation.degenerator import load_static_file
from methods_module.linear_regression import LinearRegression, LearningRateScheduling
from methods_module.polynomial_regression import PolynomialRegression
from visualisation_module.visualisation import *

if __name__ == '__main__':
    data = load_static_file("noisy_1d.json")

    # X = np.array([[1], [1.5], [2], [2], [3], [3.3], [4], [5]])
    # y = np.array([0.9, 1.4, 2.1, 2, 3.1, 3.25, 4.2, 4.9])

    points = data['points']
    values = data['values']
    X = np.array(points)
    y = np.array(values)

    lr = LinearRegression(
        X, y,
        epochs=1000,
        learning_rate=0.005,
        batch_size=5,
        scheduling=LearningRateScheduling.STEPPED
    )
    lr.fit()
    print(lr.get_weights())
    print(lr.get_bias())
    print(lr.predict(X[0]))
    # print(lr.get)
    regression_result_visualisation([(points[i][0], values[i]) for i in range(len(points))], lr.get_weights()[0], lr.get_bias())

