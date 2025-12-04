import numpy as np

# ReLU Function
def relu(x):
    return np.maximum(0, x)

# ReLU Derivative
def relu_derivative(x):
    return (x > 0).astype(float)

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sigmoid Derivative
def sigmoid_derivative(z):
    return z * (1 - z)

def softmax(logits):
    """
    Compute the softmax values for a given array of scores.

    Arguments:
    logits -- numpy array of shape (num_classes,) or (num_samples, num_classes)

    Returns:
    numpy array of softmax probabilities of the same shape as logits.
    """
    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
    return exp_logits / np.sum(exp_logits)

def softmax_derivative(softmax_values):
    """
    Compute the derivative of the softmax function.

    Arguments:
    softmax_values -- numpy array of softmax probabilities

    Returns:
    Jacobian matrix of the softmax output.
    """
    S = np.diag(softmax_values)  # Diagonal matrix
    outer = np.outer(softmax_values, softmax_values)  # Outer product
    return S - outer  # Jacobian
