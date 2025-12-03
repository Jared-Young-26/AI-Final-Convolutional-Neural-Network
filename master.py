import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import datasets
from ann import train_network, predict_classes, save_model, load_model
from extra import extract_edge_segment_features, compute_edges, segment_edge_image


# ------------------------------------------------------
# 1. VISUALIZATION UTILITIES
# ------------------------------------------------------

def show_original(img):
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

def show_edges(img):
    edges = compute_edges(img)
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Map (Sobel)")
    plt.axis("off")
    plt.show()

def show_grid_segments(edges, grid_rows=4, grid_cols=4):
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    ax.imshow(edges, cmap="gray")
    ax.set_title("Grid Segmentation Overlay")

    H, W = edges.shape
    seg_h = H // grid_rows
    seg_w = W // grid_cols

    # draw grid lines
    for r in range(grid_rows+1):
        ax.axhline(r*seg_h, color="red", linewidth=0.7)
    for c in range(grid_cols+1):
        ax.axvline(c*seg_w, color="red", linewidth=0.7)

    plt.axis("off")
    plt.show()


def visualize_feature_vector(features):
    plt.bar(range(len(features)), features)
    plt.title("Feature Vector (Edge Densities)")
    plt.xlabel("Segment Index")
    plt.ylabel("Density")
    plt.show()


# ------------------------------------------------------
# 2. CONVERT LABELS TO ONE-HOT
# ------------------------------------------------------

def to_one_hot(y, num_classes):
    N = len(y)
    out = np.zeros((N, num_classes), dtype=np.float32)
    for i, label in enumerate(y):
        out[i, label] = 1.0
    return out


# ------------------------------------------------------
# 3. LOAD & PREPROCESS MNIST
# ------------------------------------------------------

def load_and_extract(grid_rows=8, grid_cols=8):
    print("Loading MNIST from TensorFlow...")
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    print("Normalizing...")
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    print("Extracting edge-segment features...")
    X_train = []
    X_test = []

    for img in x_train:
        feats = extract_edge_segment_features(img, grid_rows, grid_cols)
        X_train.append(feats)

    for img in x_test:
        feats = extract_edge_segment_features(img, grid_rows, grid_cols)
        X_test.append(feats)

    X_train = np.array(X_train)
    X_test  = np.array(X_test)

    print("Feature shape:", X_train.shape)
    return (x_train, y_train), (x_test, y_test), X_train, X_test


# ------------------------------------------------------
# 4. TRAIN CUSTOM ANN
# ------------------------------------------------------

def run_custom_ann(X_train, y_train, X_test, y_test):
    num_classes = 10
    y_train_oh = to_one_hot(y_train, num_classes)
    y_test_oh  = to_one_hot(y_test,  num_classes)

    print("\nTraining Custom ANN...")
    weights, biases, log = train_network(
        X_train, y_train_oh,
        hidden_layers=[64, 32, 16],       # You can adjust
        num_outputs=num_classes,
        learning_rate=0.0005,
        bias_value=0.0,
        max_epochs=1000
    )

    print("\nSample Training Log:")
    for line in log[:10]:
        print(line)

    preds_train = predict_classes(X_train, weights, biases)
    preds_test  = predict_classes(X_test,  weights, biases)

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
    Runs a simple CNN using TensorFlow for benchmarking

    Example Load Model:
    cnn_model = tf.keras.models.load_model("tf_cnn_model.keras")

    """
    print("\nRunning TensorFlow CNN Benchmark...")
    
    x_train_exp = np.expand_dims(x_train, -1)
    x_test_exp  = np.expand_dims(x_test,  -1)

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

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train_exp, y_train, epochs=3, batch_size=128, verbose=1)

    test_loss, acc = model.evaluate(x_test_exp, y_test, verbose=2)
    print(f"\nTF CNN Test Accuracy: {acc:.4f}")

    return model


# ------------------------------------------------------
# 6. FULL PIPELINE EXECUTION
# ------------------------------------------------------

if __name__ == "__main__":
    # --- load + feature extraction ---
    (raw_train, y_train), (raw_test, y_test), X_train, X_test = load_and_extract()

    # --- custom ANN ---
    preds_test, weights, biases = run_custom_ann(X_train, y_train, X_test, y_test)

    save_model("custom_ann_model.npz", weights, biases)

    # --- cnn benchmark ---
    cnn_model = run_tf_cnn(raw_train, y_train, raw_test, y_test)

    cnn_model.save("tf_cnn_model.keras")

    # --- visualization example ---
    idx = 0
    print("\nVisualizing Example Image...")

    img = raw_test[idx]
    edges = compute_edges(img)

    show_original(img)
    show_edges(img)
    show_grid_segments(edges)
    visualize_feature_vector(X_test[idx])

    print(f"True label: {y_test[idx]}")
    print(f"Custom ANN prediction: {preds_test[idx]}")
    print("TF CNN prediction:", 
          np.argmax(cnn_model.predict(np.expand_dims(img, (0, -1))), axis=1)[0])
