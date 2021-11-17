import numpy as np
from utils import *
from logreg import *
from naivebayes import *
from knn import *
from svm import *
from decisiontree import *
from randomforest import *
from ann import *

"""
UCI Heart Disease Data Set: https://www.kaggle.com/fedesoriano/heart-failure-prediction

12 attributes

0   Age: age of the patient [years]
1   Sex: sex of the patient 
         [M: Male, 
         F: Female]
2   ChestPainType: chest pain type 
        [TA: Typical Angina, 
        ATA: Atypical Angina, 
        NAP: Non-Anginal Pain, 
        ASY: Asymptomatic]
3   RestingBP: resting blood pressure [mm Hg]
4   Cholesterol: serum cholesterol [mm/dl]
5   FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
6   RestingECG: resting electrocardiogram results  
        [Normal: Normal, 
        ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 
        LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
7   MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
8   ExerciseAngina: exercise-induced angina 
         [Y: Yes, 
         N: No]
9   Oldpeak: oldpeak = ST [Numeric value measured in depression]
10  ST_Slope: the slope of the peak exercise ST segment 
         [Up: upsloping, 
         Flat: flat, 
         Down: downsloping]

11  HeartDisease: output class [1: heart disease, 0: Normal]
"""

#TODO: visualize dataset in 2D space using PCA and dimension reduction
#TODO: combine all algorithms into a single file

x, y = load_dataset("heart_processed.csv")

# five-fold cross validation
fold = 5
repeat = 100

# Logistic Regression Model
# penalty = 'l2'
#
# logreg = Logistic_Regression(x, y, penalty=penalty, fold=fold, repeat=repeat)
#
# print("The accuracy of the Logistic Regression classifier was {}%.".format(round(logreg * 100, 2)))
#
# # Naive Bayes (Gaussian) Model
nb = NaiveBayes(x, y, fold, repeat)

print("The accuracy of the Naive Bayes classifier was {}%.".format(round(nb * 100, 2)))
#
# # k-Nearest Neighbours Classifier
# neighbours = [10, 13, 15]     # based off the sqrt(n) rule of thumb
#
# knn = KNN_Clasifier(x, y, neighbours, fold, repeat)
#
# for i, neighbour in enumerate(neighbours):
#     print("The accuracy of the {}-Nearest Neighbours classifier was {}%.".format(neighbour, round(knn[i] * 100, 2)))
#
# # Support Vector Machine Classifier
# reg = [1.0, 2.0, 3.0, 4.0, 5.0]
# kernel = ['linear', 'rbf']
# gamma = 'scale'
#
# svm = SVM_Classifier(x, y, reg, kernel, gamma, fold, repeat)
#
# for i, c in enumerate(reg):
#     print("The accuracy of the {} SVM classifier with c={} was {}%.".format(kernel, c, round(svm[i] * 100, 2)))
#
# # Decision Tree Classifier
# max_depth = None
#
# dtree = DecisionTree(x, y, max_depth, fold, repeat)
#
# print("The accuracy of the Decision Tree classifier was {}%.".format(round(dtree * 100, 2)))
#
# # Random Forest Classifier
# estimators = 100
#
# randforest = RandomForest(x, y, estimators, fold, repeat)
#
# print("The accuracy of the Random Forest classifier was {}%.".format(round(randforest * 100, 2)))
#
# # Artificial Neural Network
# layers = (100,)  # must be tuple
# activation = 'relu'
# solver = 'adam'
# alpha = 0.0001
# scales = ['std', 'norm', 'none']
#
# ann = ANN(x, y, layers, activation, solver, alpha, fold, repeat, scales)
#
# for i, score in enumerate(ann):
#     print("The accuracy of the Artificial Neural Network with {} scaling was {}%."
#           .format(scales[i], round(score * 100, 2)))

