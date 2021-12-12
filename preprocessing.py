import pandas as pd


def main(fname='heart.csv'):

    data = pd.read_csv('heart.csv')

    encoder = {"Sex": {"M": 0, "F": 1},
               "ExerciseAngina": {"N": 0, "Y": 1}
               }

    data = data.replace(encoder)

    chestPain = pd.get_dummies(data.ChestPainType, prefix='CPT')
    restingECG = pd.get_dummies(data.RestingECG, prefix='rECG')
    stSlope = pd.get_dummies(data.ST_Slope, prefix='stSlope')

    colNames = ['ChestPainType', 'RestingECG', 'ST_Slope']

    colName = colNames[0]
    ind = data.columns.get_loc(colName)
    for col in chestPain.columns:
        data.insert(loc=ind, column=str(col), value=chestPain[col])
        ind += 1

    colName = colNames[1]
    ind = data.columns.get_loc(colName)
    for col in restingECG.columns:
        data.insert(loc=ind, column=str(col), value=restingECG[col])
        ind += 1

    colName = colNames[2]
    ind = data.columns.get_loc(colName)
    for col in stSlope.columns:
        data.insert(loc=ind, column=str(col), value=stSlope[col])
        ind += 1

    data = data.drop(colNames, axis=1)

    data.to_csv('heart_encoded.csv', header=False, encoding='utf-8', index=False)


if __name__ == '__main__':
    main()