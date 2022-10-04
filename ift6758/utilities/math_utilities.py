import numpy as np

def compute_2d_euclidean_distance(x1, y1, x2, y2):
    """
    Compute 2D Euclidean distance between (x1, y1) and (x2, y2) given by formula sqrt((x2 - x1)² + (y2 - y1)²)

    Args:
        x1: X coordinate of point 1, numeric or series
        y1: Y coordinate of point 1, numeric or series
        x2: X coordinate of point 2, numeric or series
        y2: Y coordinate of point 2, numeric or series

    Returns:
        2D Euclidean distance between (x1, y1) and (x2, y2) or of a series of coordinates

    """
    return np.sqrt(np.add(np.square(np.subtract(x2, x1)), np.square(np.subtract(y2, y1))))
