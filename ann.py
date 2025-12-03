import numpy as np

# ===== Activation Functions =====
def sigmoid(x):
    """
    DESCRIPTION:
        Computes the sigmoid activation function used in neural networks.
        This function squashes any real-valued input into the range (0, 1),
        making it useful for hidden-layer activation and probability-like outputs.

    INPUT:
        x : numpy array
            A vector or matrix of real values.

    PROCESSING:
        * Clamp extremely large/small values using np.clip to prevent overflow in exp().
        * Compute sigmoid element-wise using the formula 1 / (1 + exp(-x)).

    OUTPUT:
        numpy array
            The transformed input, with every element mapped into (0, 1).
    """
    x = np.clip(x, -60, 60)  # Prevent overflow in exp() for large magnitude values
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    DESCRIPTION:
        Computes the derivative of the sigmoid activation function.
        This is required during backpropagation for adjusting weights.

    INPUT:
        x : numpy array
            The previously-computed sigmoid activations.

    PROCESSING:
        * Apply the derivative formula: sigmoid(x) * (1 - sigmoid(x)).
        * Here, `x` is already assumed to be sigmoid(x), not raw pre-activation input.

    OUTPUT:
        numpy array
            Element-wise derivative of the sigmoid values.
    """
    return x * (1.0 - x)  # Efficient derivative using stored activation values


def tanh_derivative(z):
    """
    DESCRIPTION:
        Computes the derivative of the hyperbolic tangent activation function.
        tanh outputs values in the range (-1, 1), often leading to better gradients
        than sigmoid for deep networks.

    INPUT:
        z : numpy array
            The tanh activation output values, NOT the raw pre-activation signal.

    PROCESSING:
        * Apply derivative formula: 1 - tanh(z)^2
        * Since z is assumed to already be tanh(z), we use z directly in the formula.

    OUTPUT:
        numpy array
            Element-wise derivative values for tanh.
    """
    return 1.0 - z**2  # Standard tanh derivative


def softmax(z):
    """
    DESCRIPTION:
        Converts raw logits (scores) into normalized probabilities.
        Used exclusively in the output layer for multi-class classification.

    INPUT:
        z : numpy array (N, K)
            N = batch size, K = number of classes.

    PROCESSING:
        * Stabilize values by subtracting row-wise max to avoid overflow.
        * Exponentiate each element.
        * Normalize each row so values sum to 1.

    OUTPUT:
        numpy array (N, K)
            Row-wise probability distributions.
    """
    z = z - np.max(z, axis=1, keepdims=True)  # Stability trick
    exp_z = np.exp(z)   # Exponentiate logits
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)   # Normalize rows


def cross_entropy(y_true, y_pred):
    """
    DESCRIPTION:
        Computes the cross-entropy loss between predicted and true class probabilities.
        This is the standard loss function for multi-class classification.

    INPUT:
        y_true : numpy array (N, K)
            One-hot encoded labels.

        y_pred : numpy array (N, K)
            Softmax output probabilities.

    PROCESSING:
        * Clamp predictions to avoid log(0) numerical errors.
        * Apply cross-entropy formula:  -sum(y_true * log(y_pred)) averaged over samples.

    OUTPUT:
        float
            The mean cross-entropy loss for the batch.
    """
    eps = 1e-12    # Numerical stability epsilon
    y_pred = np.clip(y_pred, eps, 1 - eps)    # Prevent log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def relu(z):
    """
    DESCRIPTION:
        Applies the Rectified Linear Unit (ReLU) activation function.
        ReLU sets all negative values to zero, helping networks avoid saturation.

    INPUT:
        z : numpy array
            Pre-activation inputs to a layer.

    PROCESSING:
        * Replace negative elements with 0.
        * Preserve positive elements exactly.

    OUTPUT:
        numpy array
            ReLU-activated values, same shape as input.
    """
    return np.maximum(0, z)  # Fast vectorized ReLU


def relu_derivative(z):
    """
    DESCRIPTION:
        Computes the derivative of the ReLU activation function,
        needed during backpropagation.

    INPUT:
        z : numpy array
            The pre-activation input values for which ReLU was applied.

    PROCESSING:
        * Derivative is 1 for z > 0, and 0 for z <= 0.
        * Implemented via boolean mask cast to float.

    OUTPUT:
        numpy array
            Element-wise derivative of ReLU, same shape as input.
    """
    return (z > 0).astype(float)  # Boolean mask → float for gradients


# ===== Initialization =====

def create_random_matrix(rows, cols):
    """
    DESCRIPTION:
        Creates a weight matrix with small random values. This ensures the neural
        network starts with diverse parameters instead of all zeros, which would
        prevent learning.

    INPUT:
        rows (int): Number of input units to this layer.
        cols (int): Number of output units from this layer.

    PROCESSING:
        * Compute a Xavier-style scaling limit based on layer size.
        * Generate a matrix of uniformly distributed random values within that range.

    OUTPUT:
        numpy.ndarray of shape (rows, cols):
            Randomly initialized weight matrix used by the network.
    """
    # Compute Xavier initialization limit (helps stable gradient flow)
    limit = np.sqrt(6 / (rows + cols))

    # Return weights sampled uniformly from [-limit, +limit]
    return np.random.uniform(-limit, limit, (rows, cols))


def initialize_network(num_inputs, hidden_layers, num_outputs, bias_value):
    """
    DESCRIPTION:
        Builds the full set of weight matrices and bias vectors for the
        neural network based on the desired layer structure.

    INPUT:
        num_inputs   (int): Number of features in the input vector.
        hidden_layers (list of int): Units in each hidden layer.
        num_outputs  (int): Number of output classes.
        bias_value   (float): Initial constant value for all biases.

    PROCESSING:
        * Build a list defining layer sizes: [input → hidden(s) → output].
        * For each layer pair, create a random weight matrix.
        * For each layer, create a bias vector initialized to a user-supplied constant.

    OUTPUT:
        weights (list of np.ndarray): Weight matrices between each layer.
        biases  (list of np.ndarray): Bias vectors for each layer.
    """
    # Construct full layer structure, e.g. [64, 128, 64, 10]
    layer_structure = [num_inputs] + hidden_layers + [num_outputs]

    # Create weight matrix for each pair of adjacent layers
    weights = [
        create_random_matrix(layer_structure[i], layer_structure[i + 1])
        for i in range(len(layer_structure) - 1)
    ]

    # Create bias vector for each layer (except input)
    biases = [
        np.full((1, layer_structure[i + 1]), bias_value)
        for i in range(len(layer_structure) - 1)
    ]

    return weights, biases


# ===== Forward / Backward =====

def forward_pass(X, weights, biases):
    """
    DESCRIPTION:
        Passes input data forward through each layer of the neural network,
        computing activations step-by-step.

    INPUT:
        X (numpy.ndarray): Input batch of shape (N, num_features).
        weights (list): Weight matrices for each layer.
        biases  (list): Bias vectors for each layer.

    PROCESSING:
        * Start with the raw input.
        * For each layer:
            - Compute z = XW + b
            - Apply activation:
                ReLU for hidden layers
                Softmax for output layer
        * Store each layer's output to support backpropagation later.

    OUTPUT:
        activations (list of np.ndarray):
            List of activation outputs for every layer from input → output.
    """
    activations = [X]  # Store input as layer 0

    # Forward compute each layer’s activation
    for i in range(len(weights)):
        # Linear transform (z = a_prev W + b)
        z = np.dot(activations[-1], weights[i]) + biases[i]

        # Hidden layers → ReLU, Output → Softmax
        if i == len(weights) - 1:
            a = softmax(z)
        else:
            a = relu(z)

        activations.append(a)

    return activations


def backpropagate(weights, biases, activations, y, learning_rate):
    """
    DESCRIPTION:
        Computes the gradients for all weights and biases based on prediction error,
        then updates the parameters to reduce future error.

    INPUT:
        weights (list): Current weight matrices.
        biases  (list): Current bias vectors.
        activations (list): Output from each layer of forward pass.
        y (numpy.ndarray): True one-hot labels for the batch.
        learning_rate (float): Step size for gradient descent.

    PROCESSING:
        * Compute output layer delta = predicted - true.
        * Backpropagate this error through hidden layers using:
            - Matrix multiplication with next layer weights
            - ReLU derivative to scale gradient
        * Reverse deltas so they align with layer order.
        * Compute gradients for every layer:
            - grad_W = a_prevᵀ · delta
            - grad_b = sum(delta)
        * Apply small L2 regularization to reduce overfitting.
        * Update all weights and biases in place.

    OUTPUT:
        loss (float): Cross-entropy loss for the current batch.
        weights (list): Updated weight matrices.
        biases  (list): Updated bias vectors.
    """
    # Compute delta for output layer (softmax - one-hot labels)
    delta = activations[-1] - y
    deltas = [delta]

    # Backpropagate through hidden layers
    for i in reversed(range(len(weights) - 1)):
        prev = deltas[-1]

        # Propagate backward using Wᵀ
        back = np.dot(prev, weights[i + 1].T)

        # Scale by derivative of activation function (ReLU)
        delta = back * relu_derivative(activations[i + 1])

        deltas.append(delta)

    # Reverse to match forward layer ordering
    deltas.reverse()

    # Update all weights and biases
    for i in range(len(weights)):
        grad_w = np.dot(activations[i].T, deltas[i])
        grad_b = np.sum(deltas[i], axis=0, keepdims=True)

        # Apply gradient descent with small L2 penalty
        weights[i] -= learning_rate * (grad_w + 0.00001 * weights[i])
        biases[i]  -= learning_rate * grad_b

    # Compute final batch loss
    loss = cross_entropy(y, activations[-1])

    return loss, weights, biases

# ===== Training / Prediction =====
def train_network(
    X, y, hidden_layers, num_outputs,
    learning_rate=0.01, bias_value=0.1,
    max_epochs=20, batch_size=128
):
    """
    DESCRIPTION:
        Trains the custom Artificial Neural Network (ANN) using mini-batch
        gradient descent. This function coordinates dataset preprocessing,
        batch formation, forward propagation, backpropagation, and weight updates.

    INPUT:
        X : ndarray (N, num_features)
            The full feature dataset.
        y : ndarray (N, num_classes)
            One-hot encoded labels for each sample.
        hidden_layers : list[int]
            Sizes of hidden layers used during initialization.
        num_outputs : int
            Number of output neurons (typically 10 for MNIST digits).
        learning_rate : float
            Step size for gradient descent weight updates.
        bias_value : float
            Constant initialization value for all bias vectors.
        max_epochs : int
            How many full passes over the dataset to train.
        batch_size : int
            How many samples per mini-batch.

    PROCESSING:
        * Standardize X so all features operate on comparable scales.
        * Initialize network weights & biases.
        * For each epoch:
            - Shuffle the dataset to prevent learning order bias.
            - Slice the shuffled data into mini-batches.
            - Run forward_pass() on each batch.
            - Compute gradients with backpropagate().
            - Update weights and biases for that batch.
        * Log training loss periodically for reporting.

    OUTPUT:
        weights : list[np.ndarray]
            Learned weight matrices for all layers.
        biases : list[np.ndarray]
            Learned bias vectors for all layers.
        log : list[str]
            Human-readable training progress messages.
    """

    log = []                        # Store training log messages
    num_inputs = X.shape[1]         # Feature count for initializing weights

    # ------------------------------------------------------------
    # STANDARDIZE INPUTS
    # (Ensures features have comparable scales & stabilizes training)
    # ------------------------------------------------------------
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    # ------------------------------------------------------------
    # INITIALIZE NETWORK
    # (Random Xavier weight matrices + constant bias vectors)
    # ------------------------------------------------------------
    weights, biases = initialize_network(num_inputs, hidden_layers, num_outputs, bias_value)

    N = X.shape[0]                  # Number of training samples
    indices = np.arange(N)          # Indices used for shuffling

    # ------------------------------------------------------------
    # FULL TRAINING LOOP
    # ------------------------------------------------------------
    for epoch in range(max_epochs):

        # ---- Shuffle the dataset each epoch ---------------------------------
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # ---- Mini-batch gradient descent ------------------------------------
        for start in range(0, N, batch_size):
            end = start + batch_size
            Xb = X[start:end]       # Mini-batch: features
            yb = y[start:end]       # Mini-batch: labels

            # Forward pass through all layers
            activations = forward_pass(Xb, weights, biases)

            # Backpropagation computes gradients & updates the parameters
            loss, weights, biases = backpropagate(weights, biases, activations, yb, learning_rate)

        # ---- Periodically record loss for reporting -------------------------
        if epoch % 2 == 0:
            log.append(f"Epoch {epoch+1}/{max_epochs} - Loss: {loss:.6f}")

    # Final trained weights, biases, and log output
    return weights, biases, log

def predict_proba(X, weights, biases):
    """
    DESCRIPTION:
        Produces the raw probability distribution output by the ANN
        (i.e., the softmax vector for each sample).

    INPUT:
        X : ndarray (N, num_features)
        weights : list[np.ndarray]
        biases : list[np.ndarray]

    PROCESSING:
        * Run a forward propagation pass through the network.
        * Extract only the final softmax output layer.

    OUTPUT:
        ndarray (N, num_classes)
            Probability values for each output class.
    """
    return forward_pass(X, weights, biases)[-1]

def predict_classes(X, weights, biases):
    """
    DESCRIPTION:
        Converts the softmax probability vector into a discrete class prediction
        by selecting the highest-probability index.

    INPUT:
        X : ndarray
        weights : list[np.ndarray]
        biases : list[np.ndarray]

    PROCESSING:
        * Compute output probabilities via predict_proba().
        * Argmax along axis=1 selects class with highest probability.

    OUTPUT:
        ndarray (N,)
            Integer class predictions.
    """
    probs = predict_proba(X, weights, biases)
    return np.argmax(probs, axis=1)

# ============================================================
# SAVE / LOAD MODEL
# ============================================================

def save_model(filepath, weights, biases):
    """
    DESCRIPTION:
        Saves the trained ANN parameters to disk in a compressed NPZ file.
        This allows the trained model to be reused without retraining.

    INPUT:
        filepath : str
            Where to save the model.
        weights : list[np.ndarray]
        biases : list[np.ndarray]

    PROCESSING:
        * Store each weight matrix under key "Wi".
        * Store each bias vector under key "Bi".
        * Store total number of layers for reconstruction.

    OUTPUT:
        None (writes file to disk)
    """
    np.savez(filepath,
             **{f"W{i}": w for i, w in enumerate(weights)},
             **{f"B{i}": b for i, b in enumerate(biases)},
             num_layers=len(weights))
    print(f"[ANN] Model saved to {filepath}")

def load_model(filepath):
    """
    DESCRIPTION:
        Loads ANN parameters from a saved NPZ model file
        and reconstructs the weight/bias lists.

    INPUT:
        filepath : str
            Path to the model file created with save_model().

    PROCESSING:
        * Read NPZ archive.
        * Extract per-layer weights and biases.
        * Reconstruct correctly ordered lists.

    OUTPUT:
        weights : list[np.ndarray]
        biases : list[np.ndarray]
    """
    data = np.load(filepath)
    num_layers = int(data["num_layers"])

    weights = [data[f"W{i}"] for i in range(num_layers)]
    biases  = [data[f"B{i}"] for i in range(num_layers)]

    print(f"[ANN] Model loaded from {filepath}")
    return weights, biases
