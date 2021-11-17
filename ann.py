from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import numpy as np


def ANN(x, y, layer, activation, solver, alpha, fold, repeat, scales):

    scores = []

    for scale in scales:
        if scale == 'norm':
            x_norm = MinMaxScaler().fit_transform(x)
        elif scale == 'std':
            x_norm = StandardScaler().fit_transform(x)
        else:
            x_norm = x

        clf = MLPClassifier(hidden_layer_sizes=layer, activation=activation, solver=solver, alpha=alpha, max_iter=10000)
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

