from utils import *
from algorithms import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
UCI Heart Disease Data Set: https://www.kaggle.com/fedesoriano/heart-failure-prediction

12 attributes

0  Age: age of the patient [years]
1  Sex: sex of the patient 
      [M: Male, 
      F: Female]
2  ChestPainType: chest pain type 
      [TA: Typical Angina, 
      ATA: Atypical Angina, 
      NAP: Non-Anginal Pain, 
      ASY: Asymptomatic]
3  RestingBP: resting blood pressure [mm Hg]
4  Cholesterol: serum cholesterol [mm/dl]
5  FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
6  RestingECG: resting electrocardiogram results  
      [Normal: Normal, 
      ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 
      LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
7  MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
8  ExerciseAngina: exercise-induced angina 
      [Y: Yes, 
      N: No]
9  Oldpeak: oldpeak = ST [Numeric value measured in depression]
10 ST_Slope: the slope of the peak exercise ST segment 
      [Up: upsloping, 
      Flat: flat, 
      Down: downsloping]

11  HeartDisease: output class [1: heart disease, 0: Normal]
"""

#TODO: visualize dataset in 2D space using PCA and dimension reduction

x, y = load_dataset("heart_processed.csv")

# K-Fold Cross Validation
fold = 5
repeat = 15 # repeat should be 10 for final plots
cv = RepeatedKFold(n_splits=fold, n_repeats=repeat)

#bagging_clf = BaggingClassifier(clf)
#adaboost_clf = AdaBoostClassifier(clf)

# Logistic Regression Classifier
"""
solver = 'lbfgs'
penalty = 'l2'
max_iter = 10000
C = 10
logreg_clf = LogisticRegression(solver=solver, penalty=penalty, max_iter=max_iter, C=C)

generate_clf_plot(logreg_clf, x, y, cv, "Logistic Regression", "optimal_graphs/logistic_regression")
#display_results("Logistic Regression Classifier", kfold_crossvalid_evaluation(x, y, logreg_clf, fold, repeat))
generate_clf_bagging_adaboost_plots(logreg_clf, x, y, cv, "Logistic Regression", "optimal_graphs/logistic_regression")
"""

# Naive Bayes (Gaussian) Classifier
"""
naivebayes_clf = GaussianNB()
generate_clf_plot(naivebayes_clf, x, y, cv, "Naive Bayes", "optimal_graphs/naive_bayes")
#display_results("Naive Bayes classifier", kfold_crossvalid_evaluation(x, y, naivebayes_clf, fold, repeat))
generate_clf_bagging_adaboost_plots(naivebayes_clf, x, y, cv, "Naive Bayes", "optimal_graphs/naive_bayes")
"""

# K-Nearest Neighbours Classifier
"""
#neighbours = [10, 13, 15]     # based off the sqrt(n) rule of thumb
neighbours = [5]
x_normalized = MinMaxScaler().fit_transform(x)

for neighbour in neighbours:
    KNN_clf = KNeighborsClassifier(n_neighbors=neighbour, p=1, weights='uniform')
    #generate_clf_plot(KNN_clf, x_normalized, y, cv, f"KNN at K={neighbour}", "optimal_graphs/KNN_k" + str(neighbour))
    generate_clf_bagging_adaboost_plots(KNN_clf, x_normalized, y, cv, f"KNN at K={neighbour}", "optimal_graphs/KNN_k" + str(neighbour))
    #display_results(f"KNN at K={neighbour} Classifier", kfold_crossvalid_evaluation(x_normalized, y, KNN_clf, fold, repeat))
"""

# Support Vector Machine Classifier
"""
#reg_parameters = [1.0, 2.0, 3.0, 4.0, 5.0]
reg_parameters = [1]
x_normalized = StandardScaler().fit_transform(x)
kernel = 'rbf' 
gamma = 'scale' 

for c in reg_parameters:
    SVM_clf = SVC(C=c, kernel=kernel, gamma=gamma, degree=3)
    #generate_clf_plot(                    SVM_clf, x_normalized, y, cv, f"SVM at c={c}", "optimal_graphs/SVM_c" + str(c))
    generate_clf_bagging_adaboost_plots(  SVM_clf, x_normalized, y, cv, f"SVM at c={c}", "optimal_graphs/SVM_c" + str(c))
    #display_results(f"SVM at c={c} Classifier", kfold_crossvalid_evaluation(x_normalized, y, SVM_clf, fold, repeat))
"""

# Decision Tree Classifier
"""
# Can test 1-20 for min_samples_leaf and plot results

decisiontree_clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1, ccp_alpha=0.01, criterion='gini', max_features=5)

#display_results("Decision Tree Classifier", kfold_crossvalid_evaluation(x, y, decisiontree_clf, fold, repeat))
generate_clf_plot(decisiontree_clf, x, y, cv, "Decision Tree", "optimal_graphs/decision_tree")
generate_clf_bagging_adaboost_plots(decisiontree_clf, x, y, cv, "Decision Tree", "optimal_graphs/decision_tree")
"""

# Random Forest Classifier
"""
# Can test 1-200 for estimators and plot results

randomforest_clf = RandomForestClassifier(n_estimators=50, max_depth=9, criterion='gini')

#display_results("Random Forest Classifier", kfold_crossvalid_evaluation(x, y, randomforest_clf, fold, repeat))
generate_clf_plot(decisiontree_clf, x, y, cv, "Random Forest", "optimal_graphs/random_forest")
generate_clf_bagging_adaboost_plots(decisiontree_clf, x, y, cv, "Random Forest", "optimal_graphs/random_forest")
"""

# Artificial Neural Network
"""
layers = (10, 5, 2) 
activation = 'relu'
solver = 'adam'
alpha = 0.001
#scales = ['std', 'norm', 'none']
scales = ['std']
max_iter = 15000

for scale in scales:
      if scale == 'norm':     x_norm = MinMaxScaler().fit_transform(x)
      elif scale == 'std':    x_norm = StandardScaler().fit_transform(x)
      else:                   x_norm = x
      ANN_clf = MLPClassifier(hidden_layer_sizes=layers, activation=activation, solver=solver, alpha=alpha, max_iter=max_iter, early_stopping=False)

      #generate_clf_plot(ANN_clf, x, y, cv, "ANN")
      #display_results(f"ANN Classifier with {scale} scale", kfold_crossvalid_evaluation(x_norm, y, ANN_clf, fold, repeat))
      #generate_clf_plot(                  ANN_clf, x_norm, y, cv, "ANN (10, 5, 2) relu adam", "optimal_graphs/ANN_relu_adam_10_5_2")
      #generate_clf_bagging_adaboost_plots(ANN_clf, x_norm, y, cv, "ANN (10, 5, 2) relu adam", "optimal_graphs/ANN_relu_adam_10_5_2")
      # triple plot seems to run forever, not sure why
      generate_clf_bagging_plots(ANN_clf, x_norm, y, cv, "ANN (10, 5, 2) relu adam", "optimal_graphs/ANN_relu_adam_10_5_2")
      print('done')
"""

#correlation_heat_map(x, y)