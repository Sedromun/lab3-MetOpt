import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_regression


def create(X, y):
    model = TFRegressor(X.shape[1])
    model.compile(
        optimizer=tf.optimizers.SGD(learning_rate=0.01),
        loss='mean_squared_error',
    )
    model.fit(X, y, validation_split=0.25, batch_size=64, epochs=10, verbose=False)
    return model


def predict(model, X):
    return model.predict(X).flatten()


def TFRegressor(n_features, seed=42):
    return tf.keras.Sequential([
        layers.Input((n_features,)),
        layers.Dense(
            units=1, use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=seed)
        )
    ])


# X, y = make_regression(n_samples=10, n_features=1, n_informative=10)
# model = TFRegressor(X.shape[1])
#
# model.compile(
#     optimizer=tf.optimizers.SGD(learning_rate=0.01),
#     loss='mean_squared_error',
# )
#
# model.fit(X, y, validation_split=0.25, batch_size=64, epochs=10, verbose=False)
# print(model.predict(X).flatten())
# print(y)
