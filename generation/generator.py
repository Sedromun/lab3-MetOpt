import json
from random import uniform, randint

def generate_dataset(
        dimension: int,
        size: int,
        weights_lb: float,
        weights_ub: float,
        noise: float = 0.1
):
    points = [[(i // (size ** (dimension - 1 - j))) % size for j in range(dimension)] for i in range(size ** dimension)]
    weights = [uniform(weights_lb, weights_ub) for _ in range(dimension)]
    values = [
        sum([(point[i] + (size / 2) * uniform(-noise, noise)) * weights[i] for i in range(dimension)])
        if randint(1, 15) != 15 else uniform(0, size * sum([weights[i] for i in range(dimension)]))
        for point in points]
    return points, values


def generate_to_file(
        dimension: int,
        size: int,
        weights_lb: float,
        weights_ub: float,
        noise: float = 0.1,
        filename: str = ""
):
    points, values = generate_dataset(dimension, size, weights_lb, weights_ub, noise)
    path_prefix = "./data/"
    with open(path_prefix + filename, 'w') as f:
        f.write(json.dumps({"points": points, "values": values}))


if __name__ == '__main__':
    generate_to_file(3, 20, 0.5, 7, 0.2, "easy_3d.json")
