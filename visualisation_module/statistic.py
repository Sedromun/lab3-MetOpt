import os
import time

from tabulate import tabulate
from termcolor import colored

from methods_module.linear_regression import LinearRegression, LearningRateScheduling

epochs = [10, 50, 100]  # 200, 500, 1000, 2000
learning_rates = [0.0005, 0.001, 0.005]
batch_sizes = [50, 100, 200]


def mse_color(best: float | None, result: float | str) -> str:
    if type(result) == str or best is None:
        return "red"
    diff = (result - best) / best * 100
    if diff < 0.08:
        return 'light_green'
    elif diff < 1:
        return 'light_yellow'
    elif diff < 5:
        return 'yellow'
    elif diff < 15:
        return 'light_red'
    return 'red'


def time_color(best: float | None, result: float | str) -> str:
    if type(result) == str or best is None:
        return "red"
    diff = result / best
    if diff < 2:
        return 'light_green'
    elif diff < 4:
        return 'light_yellow'
    elif diff < 16:
        return "yellow"
    elif diff < 32:
        return 'light_red'
    return 'red'


def regression_stat(X, y, clazz, args):
    os.system("cls")

    best = None
    best_time = None
    tabulates = []

    for batch_size in batch_sizes:
        lines = []
        for epoch in epochs:
            line = [epoch]
            for learning_rate in learning_rates:
                lr = clazz(
                    X, y,
                    epochs=epoch,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    **args
                )
                try:
                    start_time = time.time()
                    lr.fit()
                    time.sleep(0.05)
                    end_time = time.time()

                    work_time = 1000 * (end_time - start_time) - 50
                    line.append((lr.get_error(), work_time))
                    if best is None or lr.get_error() < best:
                        best = lr.get_error()
                except Exception:
                    line.append(("error", "inf"))
            lines.append(line)

        tabulates.append(lines)

    for lines in tabulates:
        for line in lines:
            for el in line[1:]:
                if mse_color(el[0], best) in ["yellow", "light_yellow", "light_green"] and (
                        best_time is None or el[1] < best_time):
                    best_time = el[1]

    for lines, batch_size in zip(tabulates, batch_sizes):
        print(f"Batch size: {batch_size}")
        print(tabulate(
            [[line[0]] + [colored(round(el[0]) if el[0] < 10 ** 6 else el[0], mse_color(best, el[0])) + " / " + colored(
                str(round(el[1], 2)) + "ms", time_color(best_time, el[1]))
                          for el in line[1:]] for line in lines],
            headers=["Epoch count"] + [f"Learning rate = {rate}" for rate in learning_rates],
            tablefmt="grid"
        ), end="\n\n\n")

    print(f"Best learning rate: {best}")
