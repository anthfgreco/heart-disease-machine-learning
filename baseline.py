from helper import *
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def baselinePredict(clf, x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train)
    scores = []
    scores.append(clf.score(x_train, y_train))
    scores.append(clf.score(x_test, y_test))

    return scores


def saveScores(measures, models, scores, fname):
    results = {}

    for i, model in enumerate(models):
        for j, measure in enumerate(measures):
            var = str(model + '_' + measure)
            results[var] = [round(scores[i][j] * 100, 2)]

    scores = pd.DataFrame(results)
    scores.to_csv(fname, index=False)


def main(save_params=False):
    x, y = load_dataset("heart_encoded.csv")
    x_train, x_test, y_train, y_test = split_data(x, y, split_size=0.3)
    x_train_norm, x_test_norm = MinMaxScaler().fit_transform(x_train), MinMaxScaler().fit_transform(x_test)
    x_train_std, x_test_std = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)

    lr_scores = baselinePredict(LogisticRegression(max_iter=10000), x_train_norm, x_test_norm, y_train, y_test)
    nb_scores = baselinePredict(GaussianNB(), x_train_norm, x_test_norm, y_train, y_test)
    knn_scores = baselinePredict(KNeighborsClassifier(), x_train_norm, x_test_norm, y_train, y_test)
    svm_scores = baselinePredict(SVC(), x_train_std, x_test_std, y_train, y_test)
    dtree_scores = baselinePredict(DecisionTreeClassifier(), x_train_std, x_test_std, y_train, y_test)
    rf_scores = baselinePredict(RandomForestClassifier(), x_train_std, x_test_std, y_train, y_test)
    ann_scores = baselinePredict(MLPClassifier(max_iter=10000), x_train_norm, x_test_norm, y_train, y_test)

    # Save results

    score_file = 'baselineScores.csv'
    measures = ['Train', 'Test']
    models = ['LR', 'NB', 'KNN', 'SVM', 'DTree', 'RForest', 'ANN']
    scores = [lr_scores, nb_scores, knn_scores, svm_scores, dtree_scores, rf_scores, ann_scores]

    saveScores(measures, models, scores, score_file)



if __name__ == '__main__':
    main()
