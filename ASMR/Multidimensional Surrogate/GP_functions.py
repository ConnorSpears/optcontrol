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



#Define the exponentiated quadratic kernel
def exponentiated_quadratic(xa, xb, hyperparams = {'l': 0.5, 'σp': 1}):
    """Exponentiated quadratic  with σ=1"""

    l = hyperparams['l']
    σp = hyperparams['σp']


    # L2 distance
    sq_norm = -0.5 * (scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') / l ** 2)

    return σp ** 2 * np.exp(sq_norm)

# Gaussian process posterior
def GP(X1, y1, X2, kernel_func, hyperparams = {'l': 0.5, 'σp': 1}, mean_func = lambda x: 0):
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

    # Compute mean function values for X1
    mean_X1 = np.array(mean_func(X1))


    # Compute posterior mean
    μ2 = solved @ (y1 - mean_X1)

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

    return Σ2  # covariance