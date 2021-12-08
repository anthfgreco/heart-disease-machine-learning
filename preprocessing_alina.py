import pandas as pd

data = pd.read_csv('heart.csv')

encoder = {"Sex": {"M": 0, "F": 1},
           "ChestPainType": {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3},
           "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
           "ExerciseAngina": {"N": 0, "Y": 1},
           "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2}
            }

data = data.replace(encoder)

# Cholesterol
data.to_csv('heart_processed.csv', header=False, encoding='utf-8', index=False)

# No Cholesterol
data_noC = data.drop(['Cholesterol'], axis=1)
data_noC.to_csv('heart_processed_noC.csv', header=False, encoding='utf-8', index=False)