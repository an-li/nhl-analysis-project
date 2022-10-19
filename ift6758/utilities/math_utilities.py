import numpy as np
import pandas as pd


def two_dimensional_euclidean_distance(x1, y1, x2, y2):
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


def subtract_and_align_matrices(m1: pd.DataFrame, m2: pd.DataFrame, fill_value=np.nan) -> pd.DataFrame:
    """
    Align m1 with m2's indices, then perform m1 - m2 as a matrix subtraction

    Args:
        m1: Original data frame
        m2: Data frame to subtract from original data frame

    Returns:
        m1 - m2 as a matrix subtraction, with m1 aligned to m2's indices
    """
    return np.subtract(m1.align(m2, fill_value=fill_value)[0], m2)
