import numpy as np

from .activation import *
from .opt_conv import *
from .encode import one_hot_encode
from .error import *
from .opt_pooling import maxPool2D
from .init import * 
        
def forward(X, kernel, weights):
    activations = []
        
    if (len(X.shape) == 2):
        X = X[np.newaxis, :, :]
    assert X.shape == (1, 28, 28), {X.shape}

    # Convolution and Pooling
    C1 = relu(conv2D(X, kernel))
    assert C1.shape == (2, 26, 26), f"Expected (2, 26, 26), Got {C1.shape}, Sobel Shape {kernel.shape}"

    P1, indices = maxPool2D(C1, (2, 2), (2, 2))
    assert P1.shape == (2, 13, 13)

    #Flattening
    F = P1.flatten()
    assert F.shape == (338,)
    activations.append(F)

    #Hidden Flattened Layers
    for j in range(len(weights) - 1):
        F = relu(np.dot(weights[j], activations[-1]))
        #assert F.shape == (128,)
        activations.append(F)

    #Output
    Y_pred = sigmoid(np.dot(weights[-1], activations[-1]))
    Y_pred = np.where(Y_pred > 0.85, 1.0, 0.0)
    assert Y_pred.shape == (26,), {Y_pred.shape}
    activations.append(Y_pred)

    return X, activations


def backward(X, y, weights, activations, learning_rate):
    full_deltas = []

    #Gradient of loss
    loss = cce_grad(y, activations[-1])
    assert loss.shape == (26,)

    #Change Weights Output
    dW = -1 * learning_rate * np.dot(loss.reshape(-1, 1), activations[-2].reshape(-1, 1).T)
    #assert dW.shape == (26, 128)
    full_deltas.append(dW)

    #Hidden flat layers
    for j in range(len(weights) - 1, 0, -1):
        loss = relu_derivative(activations[j]) * (np.dot(weights[j].T, loss))
        #assert loss.shape == (128,)

        dW = -1 * learning_rate * np.dot(loss.reshape(-1, 1), activations[j - 1].reshape(-1, 1).T)
        #assert dW.shape == (128, 338)
        full_deltas.append(dW)
    
    #Flat weight updates
    for i in range(len(weights)):
        weights[i] += full_deltas[len(full_deltas) - 1 - i]

    return weights


def train_model_(
    X_train, y_train, weights, kernel, epochs, learning_rate, sample_size
):
    print("==Training Model==")
    
    for it in range(epochs):
        avg_err = 0
        acc = 0
        for i, (X, y) in enumerate(zip(X_train, y_train)):
            #Forward#
            X_, activations, = forward(X, kernel, weights)

            y_true = one_hot_encode(y, 26) #Factor this out
            avg_err += cce(y_true, activations[-1])
            acc += (np.argmax(activations[-1]) == np.argmax(y_true))

            #Backward#
            weights = backward(X_, y_true, weights, activations, learning_rate)
            
            if (i >= sample_size):
                break
            
        avg_err /= (sample_size + 1)
        acc /= (sample_size + 1)
        print(f"Epoch: {it + 1} Training Error: {avg_err} Training Accuracy: {acc}")

    print("==Model Trained==\n")

    print("==Saving Weights==")
    
    np.savez('custom_cnn_model_flat_no_sigmoid.npz', kernel=kernel, W1=weights[0], W2=weights[1], W3=weights[2])

    print("==Weights Saved==\n")

    return weights


def test_model(X_test, y_test, weights, kernel):
    print("==Testing Model==")

    acc = 0
    for X, y in zip(X_test, y_test):
        #Forward#
        _, activations = forward(X, weights, kernel)

        y_true = one_hot_encode(y, 26)
        acc += (np.argmax(activations[-1]) == np.argmax(y_true))

    acc /= 14800
    print(f"Testing Accuracy: {acc}")

    print("==Model Tested==\n")


def make_prediction(X):
    data = np.load("custom_model.npz")
    kernel = data["kernel"]
    W1 = data["W1"]
    W2 = data["W2"]
    W3 = data["W3"]
    weights = [W1, W2, W3]
    data.close()

    _, output = forward(X, kernel, weights)

    return np.argmax(output[-1]), output[-1]
    

def run_model(
    X_train, y_train, X_test, y_test, epochs, learning_rate, samples
):
    print("==Initializing Model Sobel Filter with Random Weights==")

    kernel = sobel()

    W1 = he_init(128, 338)
    W2 = he_init(64, 128)
    W3 = he_init(26, 64)
    weights = [W1, W2, W3]
    print("==Model Initialized==\n")
    
    train_model_(
        X_train, y_train, 
        weights, kernel,
        epochs, learning_rate, samples
    )

    test_model(X_test, y_test, kernel, weights)
  
