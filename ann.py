import numpy as np

# ===== Activation Functions =====
def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def tanh_derivative(z):
    return 1.0 - z**2

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# ===== Initialization =====
def create_random_matrix(rows, cols):
    limit = np.sqrt(6 / (rows + cols))
    return np.random.uniform(-limit, limit, (rows, cols))

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
        if i == len(weights) - 1:
            a = softmax(z)
        else:
            a = relu(z)
        activations.append(a)
    return activations

def backpropagate(weights, biases, activations, y, learning_rate):
    # Output layer: softmax + cross-entropy
    delta = activations[-1] - y
    deltas = [delta]

    # Hidden layers (backwards)
    for i in reversed(range(len(weights) - 1)):
        prev = deltas[-1]
        back = np.dot(prev, weights[i + 1].T)
        delta = back * relu_derivative(activations[i + 1])
        deltas.append(delta)

    deltas.reverse()

    # Update weights/biases (non-destructive)
    for i in range(len(weights)):
        grad_w = np.dot(activations[i].T, deltas[i])
        grad_b = np.sum(deltas[i], axis=0, keepdims=True)

        # Use small L2 penalty
        weights[i] -= learning_rate * (grad_w + 0.00001 * weights[i])
        biases[i]  -= learning_rate * grad_b

    loss = cross_entropy(y, activations[-1])
    return loss, weights, biases

def train_network(
    X, y, hidden_layers, num_outputs,
    learning_rate=0.01, bias_value=0.1,
    max_epochs=20, batch_size=128
):
    log = []
    num_inputs = X.shape[1]

    # STANDARDIZE INPUTS
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    weights, biases = initialize_network(num_inputs, hidden_layers, num_outputs, bias_value)

    N = X.shape[0]
    indices = np.arange(N)

    for epoch in range(max_epochs):
        # Shuffle each epoch
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Mini-batches
        for start in range(0, N, batch_size):
            end = start + batch_size
            Xb = X[start:end]
            yb = y[start:end]

            activations = forward_pass(Xb, weights, biases)
            loss, weights, biases = backpropagate(weights, biases, activations, yb, learning_rate)

        if epoch % 2 == 0:
            log.append(f"Epoch {epoch+1}/{max_epochs} - Loss: {loss:.6f}")

    return weights, biases, log


def predict_proba(X, weights, biases):
    return forward_pass(X, weights, biases)[-1]

def predict_classes(X, weights, biases):
    probs = predict_proba(X, weights, biases)
    return np.argmax(probs, axis=1)

# ============================================================
# SAVE / LOAD MODEL
# ============================================================
def save_model(filepath, weights, biases):
    np.savez(filepath,
             **{f"W{i}": w for i, w in enumerate(weights)},
             **{f"B{i}": b for i, b in enumerate(biases)},
             num_layers=len(weights))
    print(f"[ANN] Model saved to {filepath}")

def load_model(filepath):
    data = np.load(filepath)
    num_layers = int(data["num_layers"])

    weights = [data[f"W{i}"] for i in range(num_layers)]
    biases  = [data[f"B{i}"] for i in range(num_layers)]
    print(f"[ANN] Model loaded from {filepath}")
    return weights, biases
