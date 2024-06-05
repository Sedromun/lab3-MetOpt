from generation.degenerator import load_static_file
from methods_module.lib_regression_methods import optimizers, LibLinearRegression
from methods_module.linear_regression import LinearRegression, LearningRateScheduling
from visualisation_module.statistic import regression_stat
from visualisation_module.visualisation import *

configs = [
    {
        "class": LinearRegression,
        "args": {"scheduling": LearningRateScheduling.STEPPED}
    },
    {
        "class": LinearRegression,
        "args": {"scheduling": LearningRateScheduling.EXPONENTIAL}
    }
] + [
    {
        "class": LibLinearRegression,
        "args": {"optimizer": name}
    } for name in optimizers
]

if __name__ == '__main__':
    data = load_static_file("noisy_1d.json")
    points = data['points']
    values = data['values']
    X = np.array(points)
    y = np.array(values)

    # print(lr.get_weights())
    # print(lr.get_bias())
    # print(lr.predict(X[0]))

    config = configs[0]
    # regression_stat(X, y, config['class'], config['args'])

    # Print result visualisation
    lr = config['class'](
        X, y,
        epochs=20,
        learning_rate=0.001,
        batch_size=20,
        **config['args'])
    lr.fit()
    regression_result_visualisation(
        [(points[i][0], values[i]) for i in range(len(points))],
        lr.get_weights()[0],
        lr.get_bias()
    )

    # Print error per iteration
    error_visualisation(lr.get_errors())


