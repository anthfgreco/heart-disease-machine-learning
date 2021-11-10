from utils import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random

data = read_csv_file("heart_processed.csv", remove_header = False, convert_to_float = True)

X = data[:, :-1]
X = add_intercept(X)
Y = data[:, -1]

splits = {0.15: "85-15", 0.20: "80-20", 0.25: "75-25", 0.30: "70-30", 0.35: "65:35"}

for split in splits.keys():
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=split, random_state=random.randint(1, 10000))
    acc = 0
    clf = KNeighborsClassifier()

    # fits, predicts, and scores the KNN model for 1000 iterations
    # note: only last set of predictions are accessible but that can be changed
    # also note: default params for the KNN model were used and we might want to change that
    # last note: we should experiment with the sklearn.cross_validation particularly the
    # cross_val_score function with shuffle=True to figure out optimal k value
    
    for i in range(1000):
        clf.fit(X, Y)
        pred = clf.predict(x_test)  # these are predictions
        acc += clf.score(x_test, y_test)

    mean_acc = acc/1000 * 100

    print("The mean accuracy of the KNN model with a {} split is {:.2f}%".format(splits[split], mean_acc))