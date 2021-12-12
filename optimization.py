import warnings
import time
from matplotlib import pyplot as plt
from helper import *
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def ParameterTuning(x_train, x_test, y_train, y_test, clf, params, title, filename):

    selector = SelectPercentile()
    pipeline = Pipeline([('selector', selector), ('clf', clf)])
    params['selector__percentile'] = [40, 50, 60, 70, 80, 90, 100]
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=random.randint(1, 100000))
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    results = GridSearchCV(pipeline, params, scoring=scoring, refit='accuracy', cv=skf, verbose=1,
                           return_train_score=True)
    results.fit(x_train, y_train)
    y_pred = results.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)
    ind = results.best_index_
    scores = [results.cv_results_['mean_train_accuracy'][ind], results.cv_results_['mean_test_accuracy'][ind],
              test_acc]
    params = results.best_params_

    cfm = plot_confusion_matrix(results, x_test, y_test, cmap='magma')
    cfm.figure_.savefig(str('graphs/cfm_' + filename))
    generate_clf_bagging_adaboost_plots(
        results, x_train, y_train, skf, title, str('graphs/' + filename))

    return results, params, scores


def LogReg(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(max_iter=10000)
    params = {'clf__penalty': ['l2', 'l1'], 'clf__C': [0.01, 0.1, 1, 10],
              'clf__solver': ['newton-cg', 'lbfgs', 'saga']}
    x_train = MinMaxScaler().fit_transform(X_train)
    x_test = MinMaxScaler().fit_transform(X_test)

    return ParameterTuning(x_train, x_test, y_train, y_test, clf, params, 'Logistic Regression', 'log_reg.png')


def NaiveBayes(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    params = {}
    x_train = MinMaxScaler().fit_transform(X_train)
    x_test = MinMaxScaler().fit_transform(X_test)

    return ParameterTuning(x_train, x_test, y_train, y_test, clf, params, 'Naive Bayes', 'naive_bayes.png')


def KNN(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier()
    params = {'clf__n_neighbors': [5, 10, 13, 20, 50], 'clf__weights': ['uniform', 'distance'], 'clf__p': [1, 2]}
    x_train = MinMaxScaler().fit_transform(X_train)
    x_test = MinMaxScaler().fit_transform(X_test)

    return ParameterTuning(x_train, x_test, y_train, y_test, clf, params, 'k-Nearest Neighbour', 'knn.png')


def SVM(X_train, X_test, y_train, y_test):
    clf = SVC()
    params = {'clf__C': [0.1, 1, 5, 10], 'clf__kernel': ['linear', 'poly', 'rbf'],
              'clf__degree': [3, 4, 5, 6], 'clf__gamma': ['scale', 'auto']}
    x_train = StandardScaler().fit_transform(X_train)
    x_test = StandardScaler().fit_transform(X_test)

    return ParameterTuning(x_train, x_test, y_train, y_test, clf, params, 'Support Vector Machine', 'svm.png')


def DTree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    params = {'clf__max_depth': [None, 9, 8, 7, 6, 5], 'clf__min_samples_leaf': [1, 10, 25, 50],
              'clf__ccp_alpha': [0, 0.05, 0.1, 0.2, 0.5]}
    x_train = StandardScaler().fit_transform(X_train)
    x_test = StandardScaler().fit_transform(X_test)

    return ParameterTuning(x_train, x_test, y_train, y_test, clf, params, 'Decision Tree', 'decision_tree.png')


def RForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier()
    params = {'clf__n_estimators': [25, 50, 100, 150], 'clf__min_samples_leaf': [1, 10, 25, 50],
              'clf__max_depth': [None, 9, 8, 7, 6, 5]}
    x_train = StandardScaler().fit_transform(X_train)
    x_test = StandardScaler().fit_transform(X_test)

    return ParameterTuning(x_train, x_test, y_train, y_test, clf, params, 'Random Forest', 'random_forest.png')


def ANN(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(max_iter=10000)
    params = {'clf__hidden_layer_sizes': [(100,), (10, 10), (10, 5, 2)],
              'clf__activation': ['logistic', 'relu'], 'clf__alpha': [0.0001, 0.001, 0.01, 0.1],
              'clf__early_stopping': [True, False]}
    x_train = MinMaxScaler().fit_transform(X_train)
    x_test = MinMaxScaler().fit_transform(X_test)

    return ParameterTuning(x_train, x_test, y_train, y_test, clf, params, 'Artificial Neural Network', 'ann.png')


def saveScores(measures, models, scores, fname):
    results = {}

    for i, model in enumerate(models):
        for j, measure in enumerate(measures):
            var = str(model + '_' + measure)
            results[var] = [round(scores[i][j] * 100, 2)]

    scores = pd.DataFrame(results)
    scores.to_csv(fname, index=False)


def saveParams(params, names):
    for i, param in enumerate(params):
        results = {}
        for p, v in param.items():
            results[p] = [v]
        results = pd.DataFrame(results)
        results.to_csv(names[i], index=False)

    return


def main(save_params=True):
    x, y = load_dataset("heart_encoded.csv")
    #correlation_heat_map(x, y)
    x_train, x_test, y_train, y_test = split_data(x, y, split_size=0.3)

    t = time.time()
    lr_params, lr_scores = LogReg(x_train, x_test, y_train, y_test)
    total = time.time() - t
    print('Logistic Regression Done! It took {} minutes to finish.'.format(round(total/60, 2)))

    t = time.time()
    nb_params, nb_scores = NaiveBayes(x_train, x_test, y_train, y_test)
    total = time.time() - t
    print(' ')
    print('Naive Bayes Done! It took {} minutes to finish.'.format(round(total/60, 2)))

    t = time.time()
    knn_params, knn_scores = KNN(x_train, x_test, y_train, y_test)
    total = time.time() - t
    print(' ')
    print('KNN Done! It took {} minutes to finish.'.format(round(total/60, 2)))

    t = time.time()
    svm_params, svm_scores = SVM(x_train, x_test, y_train, y_test)
    total = time.time() - t
    print(' ')
    print('SVM Done! It took {} minutes to finish.'.format(round(total/60, 2)))

    t = time.time()
    dtree_params, dtree_scores = DTree(x_train, x_test, y_train, y_test)
    total = time.time() - t
    print(' ')
    print('Decision Tree Done! It took {} minutes to finish.'.format(round(total/60, 2)))

    t = time.time()
    rf_params, rf_scores = RForest(x_train, x_test, y_train, y_test)
    total = time.time() - t
    print(' ')
    print('Random Forest Done! It took {} minutes to finish.'.format(round(total/60, 2)))

    t = time.time()
    ann_params, ann_scores = ANN(x_train, x_test, y_train, y_test)
    total = time.time() - t
    print(' ')
    print('Artificial Neural Network Done! It took {} minutes to finish.'.format(round(total/60, 2)))

    # Save results

    score_file = 'tunedScores.csv'
    measures = ['Train', 'CV', 'Test']
    models = ['LR', 'NB', 'KNN', 'SVM', 'DTree', 'RForest', 'ANN']
    scores = [lr_scores, nb_scores, knn_scores, svm_scores, dtree_scores, rf_scores, ann_scores]

    saveScores(measures, models, scores, score_file)

    # (Optional) Save Best Parameters

    if save_params==True:
        params = [lr_params, nb_params, knn_params, svm_params, dtree_params, rf_params, ann_params]
        names = ['lrParams.csv', 'nbParams.csv', 'knnParams.csv', 'svmParams.csv',
                'dtreeParams.csv', 'rfParams.csv', 'annParams.csv']

        saveParams(params, names)


if __name__ == '__main__':
    main()
