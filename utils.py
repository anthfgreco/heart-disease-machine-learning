import csv
import numpy as np

def add_intercept(x):
    """Add intercept to matrix x.
    Args:
        x: 2D NumPy array.
    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x
    return new_x

def read_csv_file(filename, remove_header=True, convert_to_float=False):
    """Reads csv file and returns a numpy array"""
    with open(filename, newline='') as f:
        if convert_to_float:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        else:
            reader = csv.reader(f)
        if remove_header:
            next(reader, None)
        return np.array(list(reader))

def save_to_file(nparray, filename):
    """Writes nparray to filename"""
    np.savetxt(filename, nparray, '%.1f', delimiter=",")