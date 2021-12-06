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
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns

def kfold_crossvalid_evaluation(x, y, clf, fold, repeat):
   rskf = RepeatedKFold(n_splits=fold, n_repeats=repeat)
   metric_array = np.zeros((fold * repeat, 3)) #precision, recall, f-score
   i = 0

   for train_index, test_index in rskf.split(x, y):
      x_train, x_test = x[train_index], x[test_index]
      y_train, y_test = y[train_index], y[test_index]
      clf.fit(x_train, y_train)
      y_test_pred = clf.predict(x_test)
      metric = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
      metric_array[i] = metric[:3]
      i += 1

   metrics_mean = metric_array.mean(axis=0)    #take mean of each column
   metrics_mean = metrics_mean * 100      #multiply 100 to get percent
   return metrics_mean[0], metrics_mean[1], metrics_mean[2]    #return precision mean, recall mean, f-score mean
   
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

    
    figure.savefig(filepath + "_learning_curve.png")
    #plt.show()
    plt.close()   

def generate_clf_bagging_adaboost_plots(clf, x, y, cv, alg_name, filepath):
   bagging_clf = BaggingClassifier(clf)
   adaboost_clf = AdaBoostClassifier(clf)
   title0 = f"{alg_name} Learning Curve"
   title1 = "Bagging Learning Curve"    #f"{alg_name} Bagging Classifer"
   title2 = "AdaBoost Learning Curve"   #f"{alg_name} AdaBoost Classifer"

   figure, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

   train_size, train_scores, test_scores = learning_curve(estimator=clf, X=x, y=y, cv=cv)
   train_scores_mean = np.mean(train_scores, axis=1)
   train_scores_std  = np.std(train_scores, axis=1)
   test_scores_mean  = np.mean(test_scores, axis=1)
   test_scores_std   = np.std(test_scores, axis=1)

   axes[0].set_title(title0)
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
   axes[0].plot(  train_size, 
               train_scores_mean, 
               'o-', 
               color="r",
               label="Training score")
   axes[0].plot(  train_size, 
               test_scores_mean, 
               'o-', 
               color="g",
               label="Cross-validation score")
   axes[0].legend(loc='lower right')

   train_size, train_scores, test_scores = learning_curve(estimator=bagging_clf, X=x, y=y, cv=cv)
   train_scores_mean = np.mean(train_scores, axis=1)
   train_scores_std  = np.std(train_scores, axis=1)
   test_scores_mean  = np.mean(test_scores, axis=1)
   test_scores_std   = np.std(test_scores, axis=1)

   axes[1].set_title(title1)
   axes[1].grid()
   axes[1].fill_between(train_size, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.1,
                     color="r")
   axes[1].fill_between(train_size, 
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, 
                     alpha=0.1,
                     color="g")
   axes[1].plot(  train_size, 
               train_scores_mean, 
               'o-', 
               color="r",
               label="Training score")
   axes[1].plot(  train_size, 
               test_scores_mean, 
               'o-', 
               color="g",
               label="Cross-validation score")
   axes[1].legend(loc='lower right')

   train_size, train_scores, test_scores = learning_curve(estimator=adaboost_clf, X=x, y=y, cv=cv)

   train_scores_mean = np.mean(train_scores, axis=1)
   train_scores_std  = np.std(train_scores, axis=1)
   test_scores_mean  = np.mean(test_scores, axis=1)
   test_scores_std   = np.std(test_scores, axis=1)

   axes[2].set_title(title2)
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
   axes[2].plot(  train_size, 
                  test_scores_mean, 
                  'o-', 
                  color="g",
                  label="Cross-validation score")
   axes[2].legend(loc='lower right')

   figure.savefig(filepath + "_triple_plot.png")
   #plt.show()
   plt.close() 

def display_results(title, results):
    precision, recall, fscore = results
    print(f"{title}")
    print(f"Precision:\t{precision:.2f}%")
    print(f"Recall: \t{recall:.2f}%")
    print(f"F-score:\t{fscore:.2f}%")

def correlation_heat_map(x, y):
   data = np.column_stack((x, y))
   data = data[:,1:]    #remove intercept column
   dataframe = pd.DataFrame(data, columns=["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope","HeartDisease"])

   plt.figure(figsize=(18, 15))
   sns.set_context(context="paper", font_scale=1.7)
   plt.title("Correlation Matrix")
   sns.heatmap(dataframe.corr(), annot=True, cmap='Blues')
   plt.savefig("correlation_heat_map.png")
   plt.close()
