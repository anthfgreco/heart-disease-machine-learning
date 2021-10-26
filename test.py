from utils import add_intercept
import csv
import numpy as np

"""
UCI Heart Disease Data Set: https://www.kaggle.com/fedesoriano/heart-failure-prediction

12 attributes

Age: age of the patient [years]
Sex: sex of the patient [M: Male, F: Female]
ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
RestingBP: resting blood pressure [mm Hg]
Cholesterol: serum cholesterol [mm/dl]
FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
RestingECG: resting electrocardiogram results  
   [Normal: Normal, 
   ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 
   LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
Oldpeak: oldpeak = ST [Numeric value measured in depression]
ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
HeartDisease: output class [1: heart disease, 0: Normal]
"""

with open('heart.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader, None)  #skip the first list
    data = np.array(list(reader))

X = data[:, :-1]     #get all columns except last
X = add_intercept(X) #add 1's to first column

Y = data[:, -1]      #get last column

print("X:\n", X)
print("\nY:\n", Y)