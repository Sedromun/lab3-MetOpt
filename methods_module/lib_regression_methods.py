import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

from generation.degenerator import load_static_file

optimizers = {
    'SGD': lambda x: tf.optimizers.SGD(learning_rate=x),
    'Momentum': lambda x: tf.optimizers.SGD(learning_rate=x, momentum=0.9),
    'Nesterov': lambda x: tf.optimizers.SGD(learning_rate=x, momentum=0.9, nesterov=True),
    'AdaGrad': lambda x: tf.optimizers.Adagrad(learning_rate=x),
    'RMSProp': lambda x: tf.optimizers.RMSprop(learning_rate=x),
    'Adam': lambda x: tf.optimizers.Adam(learning_rate=x)
}


class LibLinearRegression:
    def __init__(self, X, y, epochs: int, batch_size: int, learning_rate: float = 0.01, **kwargs):
        self.X = StandardScaler().fit_transform(X).astype(np.float32)
        self.y = y
        self.epochs = epochs
        self.batch_size = batch_size

        if "optimizer" in kwargs:
            self.optimizer_name = kwargs["optimizer"]
        else:
            self.optimizer_name = "SGD"
        self.optimizer = optimizers[self.optimizer_name](learning_rate)
        self.model = Sequential()
        self.model.add(Input((X.shape[1],)))
        self.model.add(Dense(1))
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def fit(self):
        self.model.fit(self.X, self.y, epochs=self.epochs, verbose=0, batch_size=self.batch_size)

    def get_error(self):
        return mean_squared_error(self.y, self.model.predict(self.X))


if __name__ == '__main__':
    data = load_static_file("noisy_1d.json")
    points = data['points']
    values = data['values']
    X = np.array(points)
    y = np.array(values)
    results = []

    for name, optimizer in optimizers.items():
        llr = LibLinearRegression(X, y, epochs=50, batch_size=20, optimizer=name)
        start_time = time.time()
        llr.fit()
        training_time = time.time() - start_time

        results.append({
            'Optimizer': name,
            'MSE': llr.get_error(),
            'Training Time (s)': training_time
        })

    results_df = pd.DataFrame(results)
    print(results_df)
