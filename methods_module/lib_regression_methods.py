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

data = load_static_file("noisy_1d.json")
points = data['points']
values = data['values']
X = np.array(points)
y = np.array(values)
# X, y = make_regression(n_samples=1000, n_features=1, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = X, X, y, y  # train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

optimizers = {
    'SGD': tf.optimizers.SGD(learning_rate=0.01),
    'Momentum': tf.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'Nesterov': tf.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    'AdaGrad': tf.optimizers.Adagrad(learning_rate=0.01),
    'RMSProp': tf.optimizers.RMSprop(learning_rate=0.01),
    'Adam': tf.optimizers.Adam(learning_rate=0.01)
}

results = []

for name, optimizer in optimizers.items():
    model = Sequential()
    model.add(Input((X_train.shape[1],)))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    start_time = time.time()
    model.fit(X_train, y_train, epochs=50, verbose=0, batch_size=20)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    results.append({
        'Optimizer': name,
        'MSE': mse,
        'Training Time (s)': training_time
    })

results_df = pd.DataFrame(results)
print(results_df)
