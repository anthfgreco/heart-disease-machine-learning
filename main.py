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
repeat = 4  # repeat should be 10 for final plots
cv = RepeatedKFold(n_splits=fold, n_repeats=repeat)

#bagging_clf = BaggingClassifier(clf)
#adaboost_clf = AdaBoostClassifier(clf)

# Logistic Regression Classifier
"""
solver = 'lbfgs'
penalty = 'l2'
max_iter = 10000
logreg_clf = LogisticRegression(solver=solver, penalty=penalty, max_iter=max_iter)

#generate_clf_plot(logreg_clf, x, y, cv, "Logistic Regression Classifier")
#display_results("Logistic Regression Classifier", kfold_crossvalid_evaluation(x, y, logreg_clf, fold, repeat))
generate_clf_bagging_adaboost_plots(logreg_clf, x, y, cv, alg_name="Logistic Regression")
"""

# Naive Bayes (Gaussian) Classifier
"""
naivebayes_clf = GaussianNB()

#generate_clf_plot(naivebayes_clf, x, y, cv, "Naive Bayes Classifier")
#display_results("Naive Bayes classifier", kfold_crossvalid_evaluation(x, y, naivebayes_clf, fold, repeat))
generate_clf_bagging_adaboost_plots(naivebayes_clf, x, y, cv, alg_name="Naive Bayes")
"""

# K-Nearest Neighbours Classifier
"""
neighbours = [10, 13, 15]     # based off the sqrt(n) rule of thumb
x_normalized = MinMaxScaler().fit_transform(x)

for neighbour in neighbours:
    KNN_clf = KNeighborsClassifier(neighbour)
    generate_clf_plot(KNN_clf, x_normalized, y, cv, f"KNN at K={neighbour} Classifier")
    #display_results(f"KNN at K={neighbour} Classifier", kfold_crossvalid_evaluation(x_normalized, y, KNN_clf, fold, repeat))
"""

# Support Vector Machine Classifier
"""
reg_parameters = [1.0, 2.0, 3.0, 4.0, 5.0]
x_normalized = StandardScaler().fit_transform(x)
# kernel -> 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
kernel = 'rbf' 
# gamma -> 'scale', 'auto'
gamma = 'scale' 

for c in reg_parameters:
    SVM_clf = SVC(C=c, kernel=kernel, gamma=gamma)
    #generate_clf_plot(SVM_clf, x_normalized, y, cv, f"SVM at c={c} Classifier")
    display_results(f"SVM at c={c} Classifier", kfold_crossvalid_evaluation(x_normalized, y, SVM_clf, fold, repeat))
"""


# Decision Tree Classifier
"""
max_depth = None
# Can test 1-20 for min_samples_leaf and plot results
min_samples_leaf = 15
decisiontree_clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

#generate_clf_plot(decisiontree_clf, x, y, cv, "Decision Tree Classifier")
#display_results("Decision Tree Classifier", kfold_crossvalid_evaluation(x, y, decisiontree_clf, fold, repeat))
generate_clf_bagging_adaboost_plots(decisiontree_clf, x, y, cv, alg_name="Decision Tree")
"""

# Random Forest Classifier
"""
# Can test 1-200 for estimators and plot results

estimators = 100
randomforest_clf = RandomForestClassifier(n_estimators=estimators)

#generate_clf_plot(randomforest_clf, x, y, cv, "Random Forest Classifier")
#display_results("Random Forest Classifier", kfold_crossvalid_evaluation(x, y, randomforest_clf, fold, repeat))
generate_clf_bagging_adaboost_plots(randomforest_clf, x, y, cv, alg_name="Random Forest")
"""

# Artificial Neural Network
"""
layers = (100,) 
activation = 'relu'
solver = 'adam'
alpha = 1e-5
#scales = ['std', 'norm', 'none']
scales = ['norm']
max_iter = 10000

for scale in scales:
      if scale == 'norm':     x_norm = MinMaxScaler().fit_transform(x)
      elif scale == 'std':    x_norm = StandardScaler().fit_transform(x)
      else:                   x_norm = x
      ANN_clf = MLPClassifier(hidden_layer_sizes=layers, activation=activation, solver=solver, alpha=alpha, max_iter=max_iter)

      #generate_clf_plot(ANN_clf, x, y, cv, "ANN Classifier")
      display_results(f"ANN Classifier with {scale} scale", kfold_crossvalid_evaluation(x_norm, y, ANN_clf, fold, repeat))
"""

#correlation_heat_map(x, y)