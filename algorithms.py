from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
import random
import numpy as np

def kfold_crossvalid_evaluation(x, y, clf, fold, repeat):
   rskf = RepeatedKFold(n_splits=fold, n_repeats=repeat, random_state=random.randint(1, 1000000))
   metrics = np.zeros((fold * repeat, 3)) #precision, recall, f-score
   i = 0

   for train_index, test_index in rskf.split(x, y):
      x_train, x_test = x[train_index], x[test_index]
      y_train, y_test = y[train_index], y[test_index]
      clf.fit(x_train, y_train)
      y_test_pred = clf.predict(x_test)
      metric = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
      metrics[i] = metric[:3]
      i += 1

   metrics_mean = metrics.mean(axis=0)    #take mean of each column
   metrics_mean = metrics_mean * 100      #multiply 100 to get percent
   return metrics_mean[0], metrics_mean[1], metrics_mean[2]    #return precision mean, recall mean, f-score mean
   

def Logistic_Regression(x, y, penalty, fold, repeat):
   clf = LogisticRegression(penalty=penalty, max_iter=10000)
   return kfold_crossvalid_evaluation(x, y, clf, fold, repeat)

def NaiveBayes(x, y, fold, repeat):
    clf = GaussianNB()
    return kfold_crossvalid_evaluation(x, y, clf, fold, repeat)

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
    metric_list = []

    for neighbour in neighbours:
        clf = KNeighborsClassifier(n_neighbors=neighbour)
        metric = kfold_crossvalid_evaluation(x_norm, y, clf, fold, repeat)
        metric_list.append(metric)

    return metric_list

def SVM_Classifier(x, y, reg, kernel, gamma, fold, repeat):
    x_norm = StandardScaler().fit_transform(x)
    scores = []

    for c in reg:
        clf = SVC(C=c, kernel=kernel, gamma=gamma)
        mean_score = kfold_crossvalid_evaluation(x_norm, y, clf, fold, repeat)
        scores.append(mean_score)

    return np.array(scores)

def DecisionTree(x, y, max_depth, fold, repeat):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    return kfold_crossvalid_evaluation(x, y, clf, fold, repeat)

def RandomForest(x, y, estimators, fold, repeat):
    clf = RandomForestClassifier(n_estimators=estimators)
    return kfold_crossvalid_evaluation(x, y, clf, fold, repeat)

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
        mean_score = kfold_crossvalid_evaluation(x_norm, y, clf, fold, repeat)

        scores.append(mean_score)

    return np.array(scores)