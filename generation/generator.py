import json
from random import uniform

from generation.degenerator import load_static_file


def generate_dataset(
        dimension: int,
        size: int,
        weights_lb: float,
        weights_ub: float,
):
    points = [[(i // (size ** (dimension - 1 - j))) % size for j in range(dimension)] for i in range(size ** dimension)]
    weights = [uniform(weights_lb, weights_ub) for _ in range(dimension)]
    values = [sum([point[i] * weights[i] * uniform(0.5, 1.5) for i in range(dimension)]) for point in points]
    return points, values


def generate_to_file(
dimension: int,
        size: int,
        weights_lb: float,
        weights_ub: float,
        filename: str,
):
    points, values = generate_dataset(dimension, size, weights_lb, weights_ub)
    path_prefix = "./data/"
    with open(path_prefix + filename, 'w') as f:
        f.write(json.dumps({"points": points, "values": values}))


if __name__ == '__main__':
    generate_to_file(2, 100, 0.5, 10, "noised_data.json")
