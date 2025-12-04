import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    return -2 * (y_true - y_pred)

def cce(y_true, y_pred):
    """
    Compute the categorical cross-entropy loss.

    Arguments:
    y_true -- numpy array of true labels (one-hot encoded)
    y_pred -- numpy array of predicted probabilities (from softmax)

    Returns:
    Scalar value of the loss.
    """
    epsilon = 1e-15  # Small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

def cce_grad(y_true, y_pred):
    """
    Compute the gradient of the categorical cross-entropy loss w.r.t predictions.

    Arguments:
    y_true -- numpy array of true labels (one-hot encoded)
    y_pred -- numpy array of predicted probabilities (from softmax)

    Returns:
    numpy array of gradients.
    """
    num_samples = y_true.shape[0]
    return (y_pred - y_true) / num_samples
