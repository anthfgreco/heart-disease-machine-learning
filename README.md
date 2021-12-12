# CPS803 Project: Predicting Cardiovascular Disease Using Machine Learning Algorithms

Our CPS803 course final project in which we use seven different classification
algorithms to see if we can predict cardiovascular disease (CVD) using eleven 
different attributes: age, sex, chest pain type, resting blood pressure, cholesterol,
fasting blood sugar, resting electrocardiogram, max heart rate, exercise angina, old
peak, and ST slope. The classification algorithms tested were logistic regression,
naive bayes, k-nearest neighbours, support vector machine, decision tree, random
forest, and artificial neural networks.

The dataset used contains 918 total entries and was retrieved from [here](https://www.kaggle.com/sanchman/heart-failure-prediction-using-pipelines).

## Project Phases:

### Preprocessing

- converted non-numeric attributes into numeric fields
- examined dataset for missing or incomplete information
- created two different preprocessing sets – one with cholesterol and one without
  * a good argument can be made to use the non-cholesterol set since
  removing the 172 relevant entries would significantly reduce dataset size
- explore the dataset
  * plot each feature's distribution w/ statistical measures

### ML Implementation

- implemented logistic regression, naive bayes, k-nearest neighbours, and SVM
- implemented decision tree, random forest, and artificial neural network
  * this can be useful for determining which kernel to use for SVM
- have two modes for each algorithm: **cross_val** for hyperparameter optimization and  
  **testing** once ideal parameters have been determined
- evaluate each model's performance using precision, recall, accuracy, 
F1, ROC AUC, and confusion matrix 
  * logistic regression: penalty term
  * naive bayes: n/a
  * knn: k-fold cross validation 
  * svm: c and kernel parameters
  * decision tree: max_depth, min_samples_leaf, and max_features
    * also look into minimal cost complexity pruning
  * random forest: n_estimators, max_depth, min_samples_leaf, and
  max_features
  * ann: alpha, and hidden_layers
    * early_stopping can also be considered


### Hyperparameter Optimization

- hyperparameter optimization 
  * **Pipeline** can be used to assemble a generalized series of steps that can
  be applied to each algorithm for cross-validation purpose while taking in 
  parameters specific to that algorithm
  * **SelectPercentile** can be used to select different percentiles of features
  for training and testing the model (e.g. compare results of 50% vs 80% 
  features) and to determine if feature importance can help improve a model
  * **GridSearchCV** can be used to specify a "grid" of parameters to test and 
  determine which yields the best results
- for each of the algorithms, tune each of the following parameters:
  * logistic regression: c, penalty type 
  * naive_bayes: n/a
  * knn: n_neighbours, weights, metric
  * svm: c, kernel, degree, gamma, coef0
  * decision tree: criterion, max_depth, min_samples_leaf, max_features
    * ccp_alpha if doing pruning
  * random forest: n_estimators, criterion, max_depth, min_samples_leaf, 
  max_features
  * artificial neural network: hidden_layers, activation, solver, alpha, learning 
  rate, early_stopping

### Analysis 

- figures/tables/chart that show the dataset as well as results
- fundamental failure analysis on the results for each algorithm
  * discuss why certain algorithms performed better than others
  * posit what can be changed or done differently to improve results obtained
  * look into ensemble methods and consider how they might help (Bagging, AdaBoost)
- talk about the bias-variance trade-off

