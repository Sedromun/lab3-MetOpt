from typing import Callable

import numpy as np

from math_module.functions import functions
from methods_module.linear_regression import LinearRegression, LearningRateScheduling
from methods_module.gradient import FunctionNonConvergence, gradient_descent
from methods_module.my_bfgs import my_bfgs
from methods_module.newton import newton
from visualisation_module.statistic import sub_stat
from visualisation_module.visualisation import *
from methods_module.scipy_methods import *
from methods_module.d1_methods import *
from methods_module.coordinate_descent import *
from random import randint as rand
from math_module.functions import functions
from tabulate import tabulate
from methods_module.my_bfgs import my_bfgs
import time

if __name__ == '__main__':
    X = np.array([[1], [1.5], [2], [2], [3], [3.3], [4], [5]])
    y = np.array([0.9, 1.4, 2.1, 2, 3.1, 3.25, 4.2, 4.9])
    lr = LinearRegression(X, y, epochs=100, learning_rate=0.1, batch_size=1,
                          scheduling=LearningRateScheduling.STEPPED)
    lr.fit()
    print(lr.get_weights())
    print(lr.get_bias())
    print(lr.predict([10]))
