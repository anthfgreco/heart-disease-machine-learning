from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random


def load_dataset(file, header=None):
    """
    Imports the dataset from the given file
    :param file: file containing dataset
    :param remove_header: removes file header (default=False)
    :param convert_to_float: converts numbers to float (default=True)
    :return: X (input) and Y (output) of data
    """
    dataset = np.array(pd.read_csv(file, header=header))

    x = dataset[:, :-1]  # get all columns except last
    y = dataset[:, -1]  # get last column

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
