import json
from random import uniform


def generate_dataset(
        dimension: int,
        size: int,
        weights_lb: float,
        weights_ub: float,
):
    points = [[(i // (size ** (dimension - 1 - j))) % size for j in range(dimension)] for i in range(size ** dimension)]
    weights = [uniform(weights_lb, weights_ub) for _ in range(dimension)]
    values = [sum([point[i] * weights[i] * uniform(0.9, 1.1) for i in range(dimension)]) for point in points]
    return points, values


if __name__ == '__main__':
    points, values = generate_dataset(1, 100, 0.5, 2)
    print(json.dumps({"points": points, "values": values}))
