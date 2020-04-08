import numpy as np

def identity(z):
    return z

def grad_identity(z):
    return np.ones(shape=z.shape)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def grad_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    out = np.exp(z)
    return out / np.sum(out, axis=0)[None,:]

def grad_softmax(z):
    return softmax(z) * (1 - softmax(z))

def tanh(z):
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

def grad_tanh(z):
    return 1 - tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def grad_relu(z):
    return (z >= 0) * 1.0