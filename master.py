import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import datasets
from ann import train_network, predict_classes, save_model, load_model
from segments import extract_edge_segment_features, compute_edges, segment_edge_image


# ------------------------------------------------------
# 1. VISUALIZATION UTILITIES
# ------------------------------------------------------

def show_original(img):
    """
    DESCRIPTION:
        Displays a raw MNIST digit (28x28 grayscale) exactly as it appears
        before any preprocessing or transformation. The purpose is to help
        visually compare the original image to the processed edge maps and
        feature vectors.

    INPUT:
        img : 2D numpy array (28x28 float or uint8)
            The original handwritten digit.

    PROCESSING:
        - Plot the 2D array as a grayscale image.
        - Remove axes so only the digit is visible.
        - Render the image immediately using plt.show().

    OUTPUT:
        None (visual side-effect — displays figure)
    """

    plt.imshow(img, cmap="gray")  # Render image in grayscale
    plt.title("Original Image")  # Give window context
    plt.axis("off")   # Hide numerical axes for clarity
    plt.show()  # Display the figure immediately


def show_edges(img):
    """
    DESCRIPTION:
        Computes a Sobel-based edge map for the given MNIST digit and displays
        the detected edges. This gives insight into how the model extracts
        the "structure" of the digit before segmentation.

    INPUT:
        img : 2D numpy array (28x28 normalized or uint8)
            The original image from which edges will be extracted.

    PROCESSING:
        - Convert the image into an edge map using the custom Sobel operator.
        - Plot the resulting binary edge image.
        - Remove axis labels for a cleaner visualization.

    OUTPUT:
        None (visual side-effect — displays figure)
    """

    edges = compute_edges(img)  # Run Sobel filter and threshold
    plt.imshow(edges, cmap="gray")  # Display binary edges
    plt.title("Edge Map (Sobel)")  # Title describes the transform
    plt.axis("off")  # Remove axes for clarity
    plt.show()   # Display the figure


def show_grid_segments(edges, grid_rows=8, grid_cols=8):
    """
    DESCRIPTION:
        Draws a visual grid overlay on top of the edge-detected image so the
        viewer can see exactly how the digit is partitioned into segments that
        feed the feature extractor.

    INPUT:
        edges      : 2D binary numpy array (28x28)
        grid_rows  : int — number of row partitions
        grid_cols  : int — number of column partitions

    PROCESSING:
        - Plot the edge map.
        - Compute the size of each grid cell.
        - Draw horizontal and vertical red grid lines.
        - Remove axes and display result.

    OUTPUT:
        None (visual side-effect — displays figure)
    """

    fig, ax = plt.subplots(1, 1, figsize=(4,4))  # Prepare figure and axis
    ax.imshow(edges, cmap="gray")     # Visualize edges
    ax.set_title("Grid Segmentation Overlay")    # Explain the visualization

    H, W = edges.shape    # Height & width of image
    seg_h = H // grid_rows  # Height of each grid cell
    seg_w = W // grid_cols   # Width of each grid cell

    # Draw horizontal grid lines across the image
    for r in range(grid_rows + 1):
        ax.axhline(r * seg_h, color="red", linewidth=0.7)

    # Draw vertical grid lines for segmentation
    for c in range(grid_cols + 1):
        ax.axvline(c * seg_w, color="red", linewidth=0.7)

    plt.axis("off")  # Remove axes
    plt.show()  # Display visualization


def visualize_feature_vector(features):
    """
    DESCRIPTION:
        Displays the extracted edge-density feature vector as a bar plot.
        Each bar corresponds to one grid cell and indicates how “edge-rich”
        that region is.

    INPUT:
        features : 1D numpy array
            The feature vector where each value is the density of edges in a
            specific image segment.

    PROCESSING:
        - Plot a simple bar chart.
        - Label axes to show index (segment ID) and density value.
        - Display the visualization.

    OUTPUT:
        None (visual side-effect — displays figure)
    """

    plt.bar(range(len(features)), features)   # Draw each density as bar
    plt.title("Feature Vector (Edge Densities)")  # Visualization title
    plt.xlabel("Segment Index")  # X-axis describes region #
    plt.ylabel("Density")  # Y-axis shows values (0–1)
    plt.show()


# ------------------------------------------------------
# 2. CONVERT LABELS TO ONE-HOT
# ------------------------------------------------------

def to_one_hot(y, num_classes):
    """
    DESCRIPTION:
        Convert integer class labels (0-9) into one-hot encoded vectors.
        This is required because the neural network outputs a probability
        distribution over classes, and one-hot vectors let us compare
        prediction vs. truth cleanly.

    INPUT:
        y (array-like)         : Vector of integer labels (length N)
        num_classes (int)      : Total number of classes (MNIST = 10)

    PROCESSING:
        - Initialize an Nxnum_classes matrix of zeros.
        - Loop through each label.
        - Set the column matching the digit's label to 1.

    OUTPUT:
        out (ndarray float32)  : One-hot encoded label matrix of shape (N, num_classes)
    """
    N = len(y)
    out = np.zeros((N, num_classes), dtype=np.float32)

    # Loop through dataset and set the true class index to 1
    for i, label in enumerate(y):
        out[i, label] = 1.0

    return out


# ------------------------------------------------------
# 3. LOAD & PREPROCESS MNIST
# ------------------------------------------------------

def load_and_extract(grid_rows=8, grid_cols=8):
    """
    DESCRIPTION:
        Load MNIST from TensorFlow, normalize pixel intensity,
        extract custom edge-segment features, and prepare the dataset
        for training the manual ANN.

    INPUT:
        grid_rows (int) : Number of vertical grid partitions per image
        grid_cols (int) : Number of horizontal grid partitions per image

    PROCESSING:
        - Load MNIST using TensorFlow utilities.
        - Normalize pixel values to the range [0,1].
        - Create empty feature lists for train/test datasets.
        - For each image:
            * Compute Sobel edges.
            * Divide the edge map into grid_rows × grid_cols cells.
            * Measure edge density in each cell.
        - Convert feature lists into numpy arrays.

    OUTPUT:
        (x_train, y_train)  : Original training images and labels
        (x_test, y_test)    : Original test images and labels
        X_train (ndarray)   : Extracted feature vectors for training set
        X_test  (ndarray)   : Extracted feature vectors for test set
    """
    print("Loading MNIST from TensorFlow...")
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    print("Normalizing...")
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    print("Extracting edge-segment features...")
    X_train = []
    X_test = []

    # ----- Extract features for training images -----
    for img in x_train:
        feats = extract_edge_segment_features(img, grid_rows, grid_cols)
        X_train.append(feats)

    # ----- Extract features for test images -----
    for img in x_test:
        feats = extract_edge_segment_features(img, grid_rows, grid_cols)
        X_test.append(feats)

    # Convert lists to arrays (shape: N × (grid_rows*grid_cols))
    X_train = np.array(X_train)
    X_test  = np.array(X_test)

    print("Feature shape:", X_train.shape)
    return (x_train, y_train), (x_test, y_test), X_train, X_test


# ------------------------------------------------------
# 4. TRAIN CUSTOM ANN
# ------------------------------------------------------

def run_custom_ann(X_train, y_train, X_test, y_test):
    """
    DESCRIPTION:
        High-level wrapper that:
            - Converts labels to one-hot vectors
            - Trains the manual ANN defined in ann.py
            - Computes final accuracy on train and test sets

    INPUT:
        X_train (ndarray) : Training feature vectors
        y_train (ndarray) : Integer training labels
        X_test  (ndarray) : Test feature vectors
        y_test  (ndarray) : Integer test labels

    PROCESSING:
        - Convert labels → one-hot encoding.
        - Call train_network() to train the ANN.
        - Generate predictions for train/test sets.
        - Compute accuracy using argmax on network outputs.
        - Print sample log lines for transparency.

    OUTPUT:
        preds_test (ndarray) : Class predictions on test set
        weights (list)       : List of trained weight matrices
        biases (list)        : List of trained bias vectors
    """
    num_classes = 10

    # Convert integer labels into one-hot vectors
    y_train_oh = to_one_hot(y_train, num_classes)
    y_test_oh  = to_one_hot(y_test,  num_classes)

    print("\nTraining Custom ANN...")

    # ----- Train the custom ANN -----
    weights, biases, log = train_network(
        X_train, y_train_oh,
        hidden_layers=[64, 32, 16],   # Adjustable ANN architecture
        num_outputs=num_classes,
        learning_rate=0.0005,
        bias_value=0.0,
        max_epochs=1000
    )

    # Print a preview of the training log for visibility
    print("\nSample Training Log:")
    for line in log[:10]:
        print(line)

    # ----- Compute train & test predictions -----
    preds_train = predict_classes(X_train, weights, biases)
    preds_test = predict_classes(X_test,  weights, biases)

    # ----- Accuracy metrics -----
    train_acc = np.mean(preds_train == y_train)
    test_acc  = np.mean(preds_test  == y_test)

    print(f"\nCustom ANN Train Acc: {train_acc:.4f}")
    print(f"Custom ANN Test Acc:  {test_acc:.4f}")

    return preds_test, weights, biases



# ------------------------------------------------------
# 5. RUN TENSORFLOW CNN FOR COMPARISON
# ------------------------------------------------------

def run_tf_cnn(x_train, y_train, x_test, y_test):
    """
    DESCRIPTION:
        Runs a small Convolutional Neural Network (CNN) using TensorFlow/Keras
        to act as a benchmark model. This provides a reference point to
        compare against the custom, manually implemented neural network.

    INPUT:
        x_train (np.ndarray): Raw MNIST training images (28×28 grayscale).
        y_train (np.ndarray): Integer class labels for training images.
        x_test  (np.ndarray): Raw MNIST test images.
        y_test  (np.ndarray): Integer class labels for test images.

    PROCESSING:
        * Expand image tensors to include a channel dimension (HxWx1).
        * Build a small sequential CNN:
            - Conv2D → ReLU activation
            - MaxPooling
            - Conv2D → ReLU activation
            - MaxPooling
            - Flatten → Dense → Dropout → Output layer
        * Compile using Adam optimizer + sparse categorical cross-entropy.
        * Train for 3 epochs.
        * Evaluate accuracy on the test set.

    OUTPUT:
        model (tf.keras.Model): The trained Keras CNN model.
    """

    print("\nRunning TensorFlow CNN Benchmark...")
    
    # Expand dims → TensorFlow expects (batch, height, width, channels)
    x_train_exp = np.expand_dims(x_train, -1)
    x_test_exp  = np.expand_dims(x_test,  -1)

    # ----------------------------------------------------------
    # Build the CNN architecture:
    #   Conv → Pool → Conv → Pool → Dense → Dropout → Softmax
    # ----------------------------------------------------------
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile with standard parameters for MNIST classification
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the CNN for a small number of epochs (fast benchmark)
    model.fit(x_train_exp, y_train, epochs=3, batch_size=128, verbose=1)

    # Evaluate generalization on the real MNIST test set
    test_loss, acc = model.evaluate(x_test_exp, y_test, verbose=2)
    print(f"\nTF CNN Test Accuracy: {acc:.4f}")

    return model


# ------------------------------------------------------
# 6. FULL PIPELINE EXECUTION
# ------------------------------------------------------

if __name__ == "__main__":
    """
    DESCRIPTION:
        Executes the entire experiment end-to-end:
        1. Loads MNIST.
        2. Extracts edge-segment features.
        3. Trains the custom ANN.
        4. Trains a TensorFlow CNN as comparison.
        5. Saves both models.
        6. Visualizes one sample prediction.

    PROCESSING:
        * Feature extraction using Sobel + grid segmentation.
        * One-hot encoding of labels for the ANN.
        * Train ANN → Save ANN weights.
        * Train CNN → Save CNN model.
        * Visualize edges + feature vector on a test sample.
        * Print predictions from both models.

    OUTPUT:
        (Printed to console)
        * ANN accuracy
        * CNN accuracy
        * Visualization of sample
        * Predictions from both models
    """

    # --- 1. Load MNIST + extract engineered features for ANN ---
    (raw_train, y_train), (raw_test, y_test), X_train, X_test = load_and_extract()

    # --- 2. Train custom ANN on engineered features ---
    preds_test, weights, biases = run_custom_ann(X_train, y_train, X_test, y_test)

    # Save trained ANN parameters
    save_model("custom_ann_model.npz", weights, biases)

    # --- 3. Train TensorFlow CNN benchmark ---
    cnn_model = run_tf_cnn(raw_train, y_train, raw_test, y_test)

    # Save CNN model to disk
    cnn_model.save("tf_cnn_model.keras")

    # --- 4. Visualize 1 sample prediction ---
    idx = 0
    print("\nVisualizing Example Image...")

    img = raw_test[idx]
    edges = compute_edges(img)

    # Visual outputs
    show_original(img)
    show_edges(img)
    show_grid_segments(edges)
    visualize_feature_vector(X_test[idx])

    # Predictions
    print(f"True label: {y_test[idx]}")
    print(f"Custom ANN prediction: {preds_test[idx]}")

    print("TF CNN prediction:", np.argmax(cnn_model.predict(np.expand_dims(img, (0, -1))), axis=1)[0])
