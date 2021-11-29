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

# K cross validation
fold = 5
repeat = 5

# Logistic Regression Model
"""
penalty = 'l2'
precision, recall, fscore = Logistic_Regression(x, y, penalty=penalty, fold=fold, repeat=repeat)
print(f"Logistic Regression classifier")
print(f"Precision:\t{precision:.2f}%")
print(f"Recall: \t{recall:.2f}%")
print(f"F-score:\t{fscore:.2f}%")
"""

# Naive Bayes (Gaussian) Model
"""
precision, recall, fscore = NaiveBayes(x, y, fold, repeat)
print(f"Naive Bayes classifier") 
print(f"Precision:\t{precision:.2f}%")
print(f"Recall: \t{recall:.2f}%")
print(f"F-score:\t{fscore:.2f}%")
"""

# k-Nearest Neighbours Classifier
"""
neighbours = [10, 13, 15]     # based off the sqrt(n) rule of thumb
knn_metrics = KNN_Clasifier(x, y, neighbours, fold, repeat)
for i, neighbour in enumerate(neighbours):
	print(f"{neighbour}-Nearest Neighbours classifier")
	print(f"Precision:\t{knn_metrics[i][0]:.2f}%")
	print(f"Recall: \t{knn_metrics[i][1]:.2f}%")
	print(f"F-score:\t{knn_metrics[i][2]:.2f}%")
"""

# Support Vector Machine Classifier
"""
reg = [1.0, 2.0, 3.0, 4.0, 5.0]
kernel = 'rbf'          #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed', default='rbf'
gamma = 'scale'
svm_metrics = SVM_Classifier(x, y, reg, kernel, gamma, fold, repeat)

for i, c in enumerate(reg):
	print(f"{kernel} SVM classifier with c={c}")
	print(f"Precision:\t{svm_metrics[i][0]:.2f}%")
	print(f"Recall: \t{svm_metrics[i][1]:.2f}%")
	print(f"F-score:\t{svm_metrics[i][2]:.2f}%")
"""

# Decision Tree Classifier
"""
max_depth = None
precision, recall, fscore = DecisionTree(x, y, max_depth, fold, repeat)
print(f"Decision Tree classifier") 
print(f"Precision:\t{precision:.2f}%")
print(f"Recall: \t{recall:.2f}%")
print(f"F-score:\t{fscore:.2f}%")
"""

# Random Forest Classifier
"""
estimators = 100
precision, recall, fscore = RandomForest(x, y, estimators, fold, repeat)
print(f"Random Forest classifier") 
print(f"Precision:\t{precision:.2f}%")
print(f"Recall: \t{recall:.2f}%")
print(f"F-score:\t{fscore:.2f}%")
"""

# Artificial Neural Network
"""
layers = (4,8)        #tuple
activation = 'relu'
solver = 'adam'
alpha = 1e-4
scales = ['std', 'norm', 'none']
ann_metrics = ANN(x, y, layers, activation, solver, alpha, fold, repeat, scales)

for i, metric in enumerate(ann_metrics):
	print(f"Artificial Neural Network with {scales[i]} scaling")
	print(f"Precision:\t{ann_metrics[i][0]:.2f}%")
	print(f"Recall: \t{ann_metrics[i][1]:.2f}%")
	print(f"F-score:\t{ann_metrics[i][2]:.2f}%")
"""

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

#correlation_heat_map(x, y)