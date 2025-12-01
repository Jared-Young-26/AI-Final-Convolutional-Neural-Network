import numpy as np

# ===== Activation Functions =====
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # x is already sigmoid(x)
    return x * (1 - x)

# ===== Initialization =====
def create_random_matrix(rows, cols):
    return np.random.uniform(-1.0, 1.0, (rows, cols))

def initialize_network(num_inputs, hidden_layers, num_outputs, bias_value):
    layer_structure = [num_inputs] + hidden_layers + [num_outputs]
    weights = [create_random_matrix(layer_structure[i], layer_structure[i + 1])
               for i in range(len(layer_structure) - 1)]
    biases = [np.full((1, layer_structure[i + 1]), bias_value)
              for i in range(len(layer_structure) - 1)]
    return weights, biases

# ===== Forward / Backward =====
def forward_pass(X, weights, biases):
    activations = [X]
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = sigmoid(z)
        activations.append(a)
    return activations

def backpropagate(weights, biases, activations, y, learning_rate):
    """
    y: (N, num_outputs) one-hot
    activations[-1]: (N, num_outputs)
    """
    error = y - activations[-1]                # (N, out)
    delta = error * sigmoid_derivative(activations[-1])
    deltas = [delta]

    # Hidden layers (backwards)
    for i in reversed(range(len(weights) - 1)):
        delta = np.dot(deltas[-1], weights[i + 1].T) * sigmoid_derivative(activations[i + 1])
        deltas.append(delta)
    deltas.reverse()

    # Update weights and biases
    for i in range(len(weights)):
        weights[i] += learning_rate * np.dot(activations[i].T, deltas[i])
        biases[i]  += learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    mse = np.mean(np.square(error))
    return mse, weights, biases

def train_network(X, y, hidden_layers, num_outputs,
                  learning_rate=0.1, bias_value=0.1,
                  max_epochs=1000, error_threshold=0.01):
    """
    X: (N, num_features)
    y: (N, num_outputs) one-hot encoding
    """
    log = []
    num_inputs = X.shape[1]
    weights, biases = initialize_network(num_inputs, hidden_layers, num_outputs, bias_value)

    for epoch in range(max_epochs):
        activations = forward_pass(X, weights, biases)
        mse, weights, biases = backpropagate(weights, biases, activations, y, learning_rate)

        if epoch % 100 == 0 or mse < error_threshold:
            log.append(f"Epoch {epoch+1}/{max_epochs} - Error: {mse:.6f}")
        if mse < error_threshold:
            break

    return weights, biases, log

def predict_proba(X, weights, biases):
    """Get raw output activations for X (N, num_outputs)."""
    output = forward_pass(X, weights, biases)[-1]
    return output  # Not rounded; used for argmax

def predict_classes(X, weights, biases):
    """Return predicted class indices using argmax."""
    probs = predict_proba(X, weights, biases)
    return np.argmax(probs, axis=1)
