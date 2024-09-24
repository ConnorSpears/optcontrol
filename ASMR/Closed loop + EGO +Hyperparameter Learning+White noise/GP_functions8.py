import scipy as scipy
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.core.base.set import RangeSet
from pyomo.dae import *
from re import I
import math
import numpy as np
import seaborn as sns
import sys
import scipy.optimize
import scipy.linalg
import scipy.spatial




#Define the exponentiated quadratic kernel
def exponentiated_quadratic(xa, xb, hyperparams = {'l': 0.5, 'σp': 1}):
    """Exponentiated quadratic  with σ=1"""

    l = hyperparams['l']
    σp = hyperparams['σp']

    # L2 distance
    sq_norm = -0.5 * (scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') / l ** 2)
   

    return σp ** 2 * np.exp(sq_norm)

# Gaussian process posterior
def GP(X1, y1, X2, error, kernel_func, hyperparams = {'l': 0.5, 'σp': 1}, mean_func = lambda x: 0):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """

    # Kernel of the observations
    Σ11 = kernel_func(X1, X1, hyperparams) + (error ** 2) * np.eye(len(X1))

    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2, hyperparams)

    # Solve
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T

    # Compute posterior mean
    μ2 = solved @ (y1 - mean_func(X1))

    # Compute posterior covariance
    Σ22 = kernel_func(X2, X2, hyperparams)
    Σ2 = Σ22 - (solved @ Σ12)

    return μ2, Σ2  # mean, covariance



# Gaussian process posterior
def VariancePosterior(X1, X2, kernel_func, hyperparams = {'l': 0.5, 'σp': 1}, mean_func = lambda x: 0):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1, hyperparams)

    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2, hyperparams)

    # Solve
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T

    # Compute posterior covariance
    Σ22 = kernel_func(X2, X2, hyperparams)
    Σ2 = Σ22 - (solved @ Σ12)

    return Σ2  # mean, covariance


def nlml(hyperparams, X, y, error, kernel_func):
    """
    Negative Log Marginal Likelihood for a Gaussian Process.
    """
    hyperparams_dict = {'l': hyperparams[0], 'σp': hyperparams[1]}
    K = kernel_func(X, X, hyperparams_dict) + (error ** 2) * np.eye(len(X))
    K_inv = np.linalg.inv(K)
    log_det_K = np.log(np.linalg.det(K))
    n = len(y)
    
    # Compute NLML
    nlml_value = 0.5 * y.T @ K_inv @ y + 0.5 * log_det_K + 0.5 * n * np.log(2 * np.pi)
    return nlml_value

def optimize_hyperparameters(X, y, error, kernel_func, initial_hyperparams=[0.5, 1]):
    """
    Optimize hyperparameters of the Gaussian Process.
    """
    result = scipy.optimize.minimize(
        nlml, 
        initial_hyperparams, 
        args=(X, y, error, kernel_func),
        method='BFGS'
    )

    return result.x
