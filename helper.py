from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns


def load_dataset(file, header=None):
    """
    Imports the dataset from the given file
    :param file: file containing dataset
    :param remove_header: removes file header (default=False)
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_size,
                                                        random_state=random.randint(1, 10000), stratify=y)

    return x_train, x_test, y_train, y_test

def generate_clf_plot(clf, x, y, cv, title, filepath):
    train_size, train_scores, test_scores = learning_curve(estimator=clf, X=x, y=y, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)

    figure, axes = plt.subplots(1, 1, figsize=(10, 10))
    plt.title(title + " Learning Curve", fontsize=17)

    axes.grid()
    axes.fill_between(train_size, 
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, 
                    alpha=0.1,
                    color="r")
    axes.fill_between(train_size, 
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, 
                    alpha=0.1,
                    color="g")
    axes.plot(  train_size, 
                train_scores_mean, 
                'o-', 
                color="r",
                label="Training score")
    axes.plot(  train_size, 
                test_scores_mean, 
                'o-', 
                color="g",
                label="Cross-validation score")
    axes.legend(loc="best")

    figure.savefig(filepath)
    plt.close()

def generate_clf_bagging_adaboost_plots(clf, x, y, cv, alg_name, filepath):
    bagging_clf = BaggingClassifier(clf)
    adaboost_clf = AdaBoostClassifier(clf)
    title0 = f"{alg_name} Learning Curve"
    title1 = "Bagging Learning Curve"  # f"{alg_name} Bagging Classifer"
    title2 = "AdaBoost Learning Curve"  # f"{alg_name} AdaBoost Classifer"

    figure, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    train_size, train_scores, test_scores = learning_curve(
        estimator=clf, X=x, y=y, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes[0].set_title(title0)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    axes[0].grid()
    axes[0].fill_between(train_size,
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std,
                        alpha=0.1,
                        color="r")
    axes[0].fill_between(train_size,
                        test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std,
                        alpha=0.1,
                        color="g")
    axes[0].plot(train_size,
                train_scores_mean,
                'o-',
                color="r",
                label="Training score")
    axes[0].plot(train_size,
                test_scores_mean,
                'o-',
                color="g",
                label="Cross-validation score")
    axes[0].legend(loc='lower right')

    train_size, train_scores, test_scores = learning_curve(
        estimator=bagging_clf, X=x, y=y, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes[1].set_title(title1)
    axes[1].set_xlabel("Training examples")
    axes[1].grid()
    axes[1].fill_between(train_size, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
    axes[1].fill_between(train_size,
                        test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std,
                        alpha=0.1,
                        color="g")
    axes[1].plot(train_size,
                train_scores_mean,
                'o-',
                color="r",
                label="Training score")
    axes[1].plot(train_size,
                test_scores_mean,
                'o-',
                color="g",
                label="Cross-validation score")
    axes[1].legend(loc='lower right')

    train_size, train_scores, test_scores = learning_curve(
        estimator=adaboost_clf, X=x, y=y, cv=cv)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)

    axes[2].set_title(title2)
    axes[2].set_xlabel("Training examples")
    axes[2].grid()
    axes[2].fill_between(train_size, 
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, 
                    alpha=0.1,
                    color="r")
    axes[2].fill_between(train_size, 
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, 
                    alpha=0.1,
                    color="g")
    axes[2].plot(  train_size, 
                train_scores_mean, 
                'o-', 
                color="r",
                label="Training score")
    axes[2].plot(train_size, 
                test_scores_mean, 
                'o-', 
                color="g",
                label="Cross-validation score")
    axes[2].legend(loc='lower right')

    figure.savefig(filepath)
    plt.close()

def correlation_heat_map(x, y):
   data = np.column_stack((x, y))
   #data = data[:,1:]    #remove intercept column
   columns = "Age  Sex  CPT_ASY  CPT_ATA  CPT_NAP  CPT_TA  RestingBP  Cholesterol  FastingBS  rECG_LVH  rECG_Normal  rECG_ST  MaxHR  ExerciseAngina  Oldpeak  stSlope_Down  stSlope_Flat  stSlope_Up HeartDisease"
   columns = columns.split()
   #print(columns)
   dataframe = pd.DataFrame(data, columns=columns)

   plt.figure(figsize=(18, 15))
   sns.set_context(context="paper", font_scale=1.7)
   plt.title("Correlation Matrix")
   sns.heatmap(dataframe.corr(), annot=False, cmap='Blues')
   plt.savefig("correlation_heat_map.png")
   plt.close()
