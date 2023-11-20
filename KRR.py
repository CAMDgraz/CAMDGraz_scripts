#!/bin/env/ python3
# -*- coding: utf-8 -*-
"""
@author: Daniel Platero Rochart [daniel.platero-rochart@medunigraz.at]
"""

import numpy as np
import pandas as pd
import joblib
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

def general_canvas(figsize, dpi):
    """
    Customization of plots

    Returns:
        None
    """
    mpl.rc('figure', figsize=figsize, dpi=dpi)
    mpl.rc('xtick', direction='in', top=False)
    mpl.rc('xtick.major', top=False)
    mpl.rc('xtick.minor', top=False)
    mpl.rc('ytick', direction='in', right=True)
    mpl.rc('ytick.major', right= False)
    mpl.rc('ytick.minor', right=False)
    mpl.rc('axes', labelsize=20)
    plt.rcParams['axes.autolimit_mode'] = 'data'
    mpl.rc('lines', linewidth=2, color='k')
    mpl.rc('font', family='monospace', size=20)
    mpl.rc('grid', alpha=0.5, color='gray', linewidth=1, linestyle='--')

    return


def read_matrix(matrix_file, process):
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
    if process == 'train':
        matrix = pd.read_csv(matrix_file, sep=',', header=None)
        Y = np.asarray(matrix.iloc[:, 0]).reshape((len(matrix), 1))
        X = np.asarray(matrix.iloc[:, 1:])
        return X, Y

    if process == 'predict':
        matrix = pd.read_csv(matrix_file, sep=',', header=None)
        X = np.asarray(matrix.iloc[:, :])
        return X


# Training --------------------------------------------------------------------
if str(sys.argv[1]) == 'train':
    print('Training using matrix: {}'.format(str(sys.argv[2])))
    # -- reading csv matrix ---------------------------------------------------
    X, Y = read_matrix(str(sys.argv[2]), 'train')

    # =========================================================================
    # Pre-process of the input data
    # =========================================================================
    # -- split of the data ----------------------------------------------------
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y,
                                                          random_state=42)
    # -- standarization -------------------------------------------------------
    X_train_s = (X_train - np.mean(X_train))/np.std(X_train)
    X_valid_s = (X_valid - np.mean(X_valid))/np.std(X_valid)

    # =========================================================================
    # Grid Search with Cross validation for hyperparameters determination
    # =========================================================================
    # -- repeated k-fold setup ------------------------------------------------
    r_kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    model = KernelRidge(kernel='rbf')

    # -- grid search with cv invluded -----------------------------------------
    parameters = {'alpha': np.logspace(-10, 1, 20),
                  'gamma': np.logspace(-10, 1, 20)}

    grid_search = GridSearchCV(estimator=model, param_grid=parameters,
                               scoring='r2', cv=r_kf, verbose=1, n_jobs=10,
                               return_train_score=True)

    grid_search.fit(X_train_s, Y_train)  # Performing GridSearchCV
    print('\nBest Estimator: \n'
          ' {}\nBest Score: '
          ' {}'.format(grid_search.best_estimator_, grid_search.best_score_))

    # =========================================================================
    # Apply model in the test validation data
    # =========================================================================
    # -- prediction -----------------------------------------------------------
    model = grid_search.best_estimator_
    model.fit(X_train_s, Y_train)
    prediction = model.predict(X_valid_s)

    # -- scoring and metrics --------------------------------------------------
    print('\nMAE: {}'.format(mean_absolute_error(Y_valid, prediction)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(Y_valid, prediction))))
    print('R2: {}'.format(r2_score(Y_valid, prediction)))

    # -- saving results -------------------------------------------------------
    joblib.dump(model, 'KRR_model.sav')

    r_prediction = np.reshape(prediction, [prediction.shape[0],])
    r_Y_valid = np.reshape(Y_valid, [Y_valid.shape[0],])
    with open('prediction.dat', 'w') as pred_out:
        pred_out.write('Prediction using model saved as'
                       ' {}\n'.format('KRR_model'))
        pred_out.write('Validation   Prediction\n')
        for real, pred in zip(r_Y_valid, r_prediction):
            pred_out.write('{:>10.2f}   {:>10.2f}\n'.format(real, pred))

    general_canvas([12, 8], 300)
    fig, ax = plt.subplots()
    ax.scatter(r_prediction, r_Y_valid, color='black', marker='x')
    ax.set_xlabel(r'Prediction')
    ax.set_ylabel(r'Validation')
    # -- linear plot ----------------------------------------------------------
    min_val = np.concatenate((r_prediction, r_Y_valid), axis=None).min()
    max_val = np.concatenate((r_prediction, r_Y_valid), axis=None).max()
    ax.plot(np.arange(min_val - 1, max_val + 2, 1),
            np.arange(min_val - 1, max_val + 2, 1), color='blue')
    ax.set_xlim(min_val - 1, max_val + 1)
    ax.set_ylim(min_val - 1, max_val + 1)
    fig.savefig('krr_train.png')


if str(sys.argv[1]) == 'predict':
    print('Predicting using model: {}'.format(str(sys.argv[2])))
    # -- Loading model and features -------------------------------------------
    model = joblib.load('KRR_model.sav')
    X = read_matrix(str(sys.argv[2]), 'predict')
    # -- Standarization -------------------------------------------------------
    X_pred_s = (X - np.mean(X))/np.std(X)

    # -- Prediction -----------------------------------------------------------
    prediction = model.predict(X_pred_s)
    joblib.dump(prediction, 'predictions.sav')
