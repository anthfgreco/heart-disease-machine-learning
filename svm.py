from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random
import numpy as np

def SVM_Classifier(x, y, reg, kernel, gamma, fold, repeat):

    scores = []
    x_norm = StandardScaler().fit_transform(x)

    for c in reg:
        clf = SVC(C=c, kernel=kernel, gamma=gamma)
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
