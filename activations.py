import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return np.where(Z > 0, 1, 0)

def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_deriv(Z):
    return sigmoid(Z) * (1.0 - sigmoid(Z))

def ReLU_Leaky(Z, alpha=0.01):
    return np.where(Z > 0, Z, Z * alpha)

def ReLU_Leaky_deriv(Z, alpha=0.01):
    return np.where(Z > 0, 1, alpha)