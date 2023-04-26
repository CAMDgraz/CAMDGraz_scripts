#!/bin/env/ python3
# -*- coding: utf-8 -*-
"""
@author: Daniel Platero Rochart [daniel.platero-rochart@medunigraz.at]
"""

import numpy as np
import pandas as pd
import joblib
import sys
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV


def read_matrix(matrix_file):
    """
    Read matrix with the independent variables in the first column

    Parameters:
        matrix_file: str
            Path to the matrix file

    Returns:
        X: np.array
            Numpy array containing the dependent variables
        Y: np.array
            Numpy array containing the independent variables
    """
    matrix = pd.read_csv(matrix_file, sep=',', header=None)
    Y = np.asarray(matrix.iloc[:, 0])
    X = np.asarray(matrix.iloc[:, 1:])
    return X, Y


# -- dummy matrix for testing -------------------------------------------------
# X = np.random.randint(100, size=(100, 200))
# Y = np.random.randint(100, size=(100,))

# -- reading csv matrix -------------------------------------------------------
matrix_file = str(sys.argv[1])
X, Y = read_matrix(matrix_file)

# =============================================================================
# Pre-process of the input data
# =============================================================================
# -- split of the data ----------------------------------------------------
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, random_state=42)

# -- standarization -------------------------------------------------------
X_train_s = (X_train - np.mean(X_train))/np.std(X_train)
X_valid_s = (X_valid - np.mean(X_valid))/np.std(X_valid)

# =============================================================================
# Grid Search with Cross validation for hyperparameters determination
# =============================================================================
# -- repeated k-fold setup ----------------------------------------------------
r_kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
model = SVR(kernel='rbf')

# -- grid search with cv invluded ---------------------------------------------
parameters = {'C': np.logspace(1, 100, 20),
              'epsilon': np.logspace(-0.001, 1, 20)}

grid_search = GridSearchCV(estimator=model, param_grid=parameters,
                           scoring='r2', cv=r_kf, verbose=1, n_jobs=10,
                           return_train_score=True)

grid_search.fit(X_train_s, Y_train)  # Performing GridSearchCV
print('Best Estimator: \n'
      ' {}\nBest Score: '
      ' {}'.format(grid_search.best_estimator_, grid_search.best_score_))

# =============================================================================
# Apply model in the test validation data
# =============================================================================
# -- prediction ---------------------------------------------------------------
model = grid_search.best_estimator_
model.fit(X_train_s, Y_train)
prediction = model.predict(X_valid_s)

# -- scoring and metrics ------------------------------------------------------
print('MAE: {}'.format(mean_absolute_error(Y_valid, prediction)))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(Y_valid, prediction))))
print('R2: {}'.format(r2_score(Y_valid, prediction)))

# -- saving the model ---------------------------------------------------------
joblib.dump(model, 'SVR_model.sav')
