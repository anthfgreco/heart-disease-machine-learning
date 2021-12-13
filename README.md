# CPS803 Project: Predicting Cardiovascular Disease Using Machine Learning Algorithms

Our CPS803 course final project in which we use seven different classification
algorithms to see if we can predict cardiovascular disease (CVD) using eleven 
different attributes: age, sex, chest pain type, resting blood pressure, cholesterol,
fasting blood sugar, resting electrocardiogram, max heart rate, exercise angina, old
peak, and ST slope. The classification algorithms tested were logistic regression,
naive bayes, k-nearest neighbours, support vector machine, decision tree, random
forest, and artificial neural networks.

The dataset used contains 918 total entries and was retrieved from [here](https://www.kaggle.com/sanchman/heart-failure-prediction-using-pipelines).

## Create and Activate Environment:

First create the environment for the project:

```
conda env create -f environment.yml
```

Then activate it:

```
conda activate cvdpred
```

## Preprocess Data:

Run the following file to preprocess data:
```
python3 preprocessing.py
```

## Create Baseline Models:

Run the following file to create baseline models:
```
python3 baseline.py
```

## Perform Hyperparamater Optimization and Apply Bagging/AdaBoost Classifier

Run the following file to run paramter tuning, apply meta-learning algorithms to optimized models, and create confusion matrices:
```
python3 optimization.py
```

## Create Plots

Use Jupyter Notebook to create a correlation heat map, compare baseline and optimized models' accuracy, and visualize classes data
```
jupyter notebook
```
Navigate to file plots.ipynb to create them. 

## Project Phases:

### Preprocessing

- converted non-numeric attributes into numeric fields
- examined dataset for missing or incomplete information

### ML Implementation

- implemented a baseline model for logistic regression, naive bayes, k-nearest neighbours, support vector machine,
decision tree, random forest, and artificial neural network

### Hyperparameter Optimization

- hyperparameter optimization tunes each of the following parameters:
  * logistic regression: c, penalty type, solver 
  * naive_bayes: n/a
  * knn: n_neighbours, weights, p
  * svm: c, kernel, degree, gamma
  * decision tree: max_depth, min_samples_leaf, ccp_alpha
  * random forest: n_estimators, max_depth, min_samples_leaf
  * artificial neural network: hidden_layers, activation, alpha, early_stopping
- used bagging and adaboost classifier to see if optimized models could be improved

