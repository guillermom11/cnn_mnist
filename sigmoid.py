import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    sigmoid_z = sigmoid(z)
    return sigmoid_z * (1 - sigmoid_z) #this is the same as Yc(t)(1-Yc(t))