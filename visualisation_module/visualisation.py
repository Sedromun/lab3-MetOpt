import matplotlib.pyplot as plt
import numpy as np


def regression_result_visualisation(
        points: list[tuple[float, float]],
        k: float,
        b: float
):
    points = np.array(sorted(points))
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    plt.scatter(list(map(lambda x: x - points[0][0], points[:, 0])),
                list(map(lambda x: x - points[0][1], points[:, 1])))
    plt.plot([0, points[-1][0] - points[0][0]],
             [b - points[0][1], k * (points[-1][0] - points[0][0]) + b - points[0][1]], color='r', markersize=3)

    # ax.plot(
    #     [x for x, _ in points],
    #     [y for _, y in points],
    # )  # Plot some data on the axes.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title(f'LINEAR REGRESSION')
    plt.show()


def error_visualisation(
        errors: list[float]
):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(
        [i for i in range(len(errors))],
        errors,
    )  # Plot some data on the axes.
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    plt.title(f'LINEAR REGRESSION')
    plt.show()
