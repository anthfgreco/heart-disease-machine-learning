import csv
import numpy as np
from utils import *
from logistic_regression import *
from sklearn.model_selection import train_test_split
import random

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

data = read_csv_file("heart_processed.csv", remove_header=False, convert_to_float=True)

X = data[:, :-1]     #get all columns except last
X = add_intercept(X) #add 1's to first column
Y = data[:, -1]      #get last column

#print("X: \n", X)
#print("Y: \n", Y)

# test_size defines the percent size of the test data. 
# For example, test_size=0.2 means the test size will be 20% of the dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=random.randint(1, 10000))

logreg_model = LogisticRegression(step_size=0.01, max_iter=100, eps=1e-5, verbose=False)
logreg_model.fit(x_train, y_train)
pred_prob = logreg_model.predict(x_test)
pred = (pred_prob > 0.5).astype(int) #convert >0.5 to 1 and <=0.5 to 0

accuracy_percent = (np.sum(pred == y_test) / len(y_test)) * 100
print(f"Accuracy: {accuracy_percent}%")