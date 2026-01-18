import os

def load_user_ignition_points(path):
    points = []
    if not os.path.exists(path):
        return points

    with open(path, "r") as f:
        for line in f:
            i, j = line.strip().split(",")
            points.append((int(i), int(j)))
    return points
