import csv
import numpy as np
from utils import *

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

data = read_csv_file("heart.csv", remove_header=True)

# Convert string data to numerical data
for i in range(len(data)):
   # M->0 and F->1
   if data[i][1] == "M": data[i][1] = 0
   if data[i][1] == "F": data[i][1] = 1

   # TA->0 ATA->1 NAP->2 ASY->3
   if data[i][2] == "TA":  data[i][2] = 0
   if data[i][2] == "ATA": data[i][2] = 1
   if data[i][2] == "NAP": data[i][2] = 2
   if data[i][2] == "ASY": data[i][2] = 3

   # Normal->0 ST->1 LVH->2
   if data[i][6] == "Normal": data[i][6] = 0
   if data[i][6] == "ST":     data[i][6] = 1
   if data[i][6] == "LVH":    data[i][6] = 2

   # N->0 Y->1
   if data[i][8] == "N": data[i][8] = 0
   if data[i][8] == "Y": data[i][8] = 1

   # Up->0 Flat->1 Down->2
   if data[i][10] == "Up":    data[i][10] = 0
   if data[i][10] == "Flat":  data[i][10] = 1
   if data[i][10] == "Down":  data[i][10] = 2

# Change string values in data to floats
data = np.array(data, dtype=np.float16)

# Delete cholestrol column, doesn't seem to have a significant impact
#data = np.delete(data, 4, 1)   

# If cholesterol value is 0, set value to the mean of the cholesterol column
# This doesn't seem to have a significant effect on the results of the algorithms
"""
data_means = data.mean(axis=0)
for i in range(len(data)):
   # Cholesterol column
   if data[i][4] == 0:
      data[i][4] = data_means[4]
"""

save_to_file(data, "heart_processed.csv")