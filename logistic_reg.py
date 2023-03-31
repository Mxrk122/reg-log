import numpy as np
from gradient import *

def sigmoid(z):
    """Función sigmoide"""
    return 1 / (1 + np.exp(-z))

def logistic_cost(theta, X, y, lmbd):
    """Función de costo de regresión logística"""
    h = sigmoid(X @ theta)
    reg = (lmbd / (2 * len(X))) * (theta[1:] ** 2).sum()
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + reg

def logistic_cost_gradient(theta, X, y, lmbd):
    """Gradiente de la función de costo de regresión logística"""
    h = sigmoid(X @ theta)
    theta_copy = theta.copy()
    theta_copy[0] = 0
    reg_prime = lmbd * theta_copy / len(X)
    return (X.T @ (h - y)) / len(X) + reg_prime





