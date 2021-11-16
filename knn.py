from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np

def KNN_Clasifier(x, y, neighbours, fold, repeat):
    """
    Implements the k-Nearest Neighbours classifier
    :param x: input
    :param y: output
    :param neighbours: number of neighbours
    :param fold: number of folds to split dataset into
    :param repeat: number of iterations
    :return: average accuracy score for given params
    """

    x_norm = MinMaxScaler().fit_transform(x)
    scores = []

    for neighbour in neighbours:
        clf = KNeighborsClassifier(n_neighbors=neighbour)
        rskf = RepeatedKFold(n_splits=fold, n_repeats=repeat, random_state=random.randint(1, 1000000))
        score = np.zeros((fold * repeat,))
        i = 0

        for train_index, test_index in rskf.split(x_norm, y):
            x_train, x_test = x_norm[train_index], x_norm[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(x_train, y_train)
            score[i] = clf.score(x_test, y_test)
            i += 1

        scores.append(score.mean())

    return np.array(scores)
