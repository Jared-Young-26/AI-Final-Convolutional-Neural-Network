import numpy as np
from tensorflow.keras import datasets

from activation import *
from opt_conv import *
from encode import one_hot_encode
from error import cce, cce_grad
from opt_pooling import maxPool2D, reverse_max2D

def print_samples(sample):
    """Used to see how data is structured"""
    RED = "\033[31m"
    RESET = "\033[0m"
    assert len(sample) == 28
    for row in sample:
        assert len(row) == 28
        for num in row:
            s = f"{num:03d}"
            if num != 0:
                print(f"{RED}{s}{RESET} ", end = '')
            else:
                print(s, end = ' ')
        print()

        
def forward(X, kernel, weights):
    activations = []
        
    X = X[np.newaxis, :, :]
    assert X.shape == (1, 28, 28)

    # Convolution and Pooling
    C1 = relu(conv2D(X, kernel))
    assert C1.shape == (2, 26, 26)

    P1, indices = maxPool2D(C1, (2, 2), (2, 2))
    assert P1.shape == (2, 13, 13)

    #Flattening
    F = P1.flatten()
    assert F.shape == (338,)
    activations.append(F)

    #Hidden Flattened Layers
    for j in range(len(weights) - 1):
        F = relu(np.dot(weights[j], activations[-1]))
        assert F.shape == (64,)
        activations.append(F)

    #Output
    Y_pred = softmax(np.dot(weights[-1], activations[-1]))
    assert Y_pred.shape == (10,)
    activations.append(Y_pred)

    return X, activations, indices, P1.shape, C1.shape


def backward(X, y, kernel, weights, activations, indices, p_shape, c_shape, learning_rate):
    full_deltas = []

    #Gradient of loss
    loss = cce_grad(y, activations[-1])
    assert loss.shape == (10,)

    #Change Weights Output
    dW = -1 * learning_rate * np.dot(loss.reshape(-1, 1), activations[-2].reshape(-1, 1).T)
    assert dW.shape == (10, 64)
    full_deltas.append(dW)

    #Hidden flat layers
    for j in range(len(weights) - 1, 0, -1):
        loss = relu_derivative(activations[j]) * (np.dot(weights[j].T, loss))
        assert loss.shape == (64,)

        dW = -1 * learning_rate * np.dot(loss.reshape(-1, 1), activations[j - 1].reshape(-1, 1).T)
        assert dW.shape == (64, 338)
        full_deltas.append(dW)

    #Loss first flattened layer
    loss = np.dot(weights[0].T, loss)
    assert loss.shape == (338,)

    #No weights here in flattening layer so no updates

    #Loss into same shape as pooling
    loss = loss.reshape(p_shape)
    assert loss.shape == (2, 13, 13)

    #No weights here either in pooling layer so no updates

    #Expand loss with reverse pooling
    loss = reverse_max2D(loss, indices, (2, 2), (2, 2), c_shape)
    loss = np.expand_dims(loss, axis=0)
    assert loss.shape == (1, 2, 26, 26)

    #Kernel updates
    dkernel = -1 * learning_rate * conv2D(X, np.flip(loss, axis=(-2, -1)))
    kernel += dkernel

    #Flat weight updates
    for j in range(len(weights)):
        weights[j] += full_deltas[len(full_deltas) - 1 - j]

    return kernel, weights


def train_model_(X_train, y_train, kernel, weights, epochs, learning_rate, sample_size):
    print("==Training Model==")

    for it in range(epochs):
        avg_err = 0
        acc = 0
        for i, (X, y) in enumerate(zip(X_train, y_train)):
            #Forward#
            X_, activations, indices, p_shape, c_shape = forward(X, kernel, weights)

            y_true = one_hot_encode(y, 10)
            avg_err += cce(y_true, activations[-1])
            acc += (np.argmax(activations[-1]) == np.argmax(y_true))

            #Backward#
            kernel, weights = backward(X_, y_true, kernel, weights, activations, indices, p_shape, c_shape, learning_rate)
            
            if (i >= sample_size):
                break
            
        avg_err /= (sample_size + 1)
        acc /= (sample_size + 1)
        print(f"Epoch: {it + 1} Training Error: {avg_err} Training Accuracy: {acc}")

    print("==Model Trained==\n")

    print("==Saving Weights==")
    
    np_ws = np.array(weights)
    np.savez('cnn.npz', kernel=kernel, weights=np_ws)

    print("==Weights Saved==")

    return kernel, weights


def test_model(X_test, y_test, kernel, weights):
    print("==Testing Model==")

    acc = 0
    for X, y in zip(X_test, y_test):
        #Forward#
        _, activations, _, _, _ = forward(X, kernel, weights)

        y_true = one_hot_encode(y, 10)
        acc += (np.argmax(activations[-1]) == np.argmax(y_true))

    acc /= 10000
    print(f"Testing Accuracy: {acc}")

    print("==Model Tested==\n")


def run_model(X_train, y_train, X_test, y_test):
    print("==Initializing Model==")

    kernel = np.random.uniform(-0.5, 0.5, (2, 1, 3, 3))

    W1 = np.random.uniform(-0.5, 0.5, (64, 338))
    W2 = np.random.uniform(-0.5, 0.5, (10, 64))
    weights = [W1, W2]

    print("==Model Initialized==\n")
    
    train_model_(x_train, y_train, kernel, weights, 5, 0.1, 60000)

    test_model(x_test, y_test, kernel, weights)

if __name__ == "__main__":
    print("==Loading Data==")
    
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data();
    x_train = x_train / 255
    x_test = x_test / 255

    print("==Data Loaded==\n")

    run_model(x_train, y_train, x_test, y_test)
    
