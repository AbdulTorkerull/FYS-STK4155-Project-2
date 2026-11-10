"""File containing simple functions such as the activation functions and their derivatives, cost functions 
and their derivatives, and the Runge function"""

import numpy as np

def identity(z):
    return z

def identity_der(z):
    return np.ones_like(z)

def ReLU(z):
    return np.where(z > 0, z, 0.0)

def ReLU_der(z):
    return np.where(z > 0, 1.0, 0.0)

def leaky_ReLU(z, a=0.01):
    return np.where(z > 0, z, a*z)

def leaky_ReLU_der(z, a=0.01):
    return np.where(z > 0, 1, a)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_der(z):
    sig = sigmoid(z)
    
    return sig*(1 - sig)

def mse(predict, target):
    return np.mean((predict - target)**2)

def mse_der(predict, target):
    return (2/predict.size)*(predict - target)

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z/np.sum(e_z, axis=1, keepdims=True)

def cross_entropy(predict, target):
    return np.mean(np.sum(-target*np.log(predict), axis=1))

def runge(x):
    return 1/(1 + 25*x**2)