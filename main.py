import csv
import numpy as np
from utils import *

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

print("X:\n", X)
#print("\nY:\n", Y)