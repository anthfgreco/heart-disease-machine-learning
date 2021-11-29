import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(file, remove_header=False, convert_to_float=True):
    """
    Imports the dataset from the given file
    :param file: file containing dataset
    :param remove_header: removes file header (default=False)
    :param convert_to_float: converts numbers to float (default=True)
    :return: X (input) and Y (output) of data
    """
    data = read_csv_file(file, remove_header=remove_header, convert_to_float=convert_to_float)

    x = data[:, :-1]  # get all columns except last
    x = add_intercept(x)  # add 1's to first column
    y = data[:, -1]  # get last column

    return x, y


def split_data(x, y, split_size, remove_header=False, convert_to_float=True):
    """
    Splits the dataset into training and test sets using the given ratio
    :param x: dataset input
    :param y: dataset output
    :param split_size: ratio of test to training set split
    :param remove_header: removes file header (default=False)
    :param convert_to_float: converts numbers to float (default=True)
    :return: training and test set
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=split_size, random_state=random.randint(1, 10000))

    return x_train, x_test, y_train, y_test


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
    np.savetxt(filename, nparray, '%.1f', delimiter=",\t")