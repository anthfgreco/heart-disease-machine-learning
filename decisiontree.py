from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
import random
import numpy as np

def DecisionTree(x, y, max_depth, fold, repeat):

    clf = DecisionTreeClassifier(max_depth=max_depth)
    rskf = RepeatedKFold(n_splits=fold, n_repeats=repeat, random_state=random.randint(1, 1000000))
    score = np.zeros((fold * repeat,))
    i = 0

    for train_index, test_index in rskf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(x_train, y_train)
        score[i] = clf.score(x_test, y_test)
        i += 1

    return score.mean()
