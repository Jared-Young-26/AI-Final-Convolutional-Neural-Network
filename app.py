import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from ann import load_model, predict_classes, predict_proba
from segments import compute_edges, extract_edge_segment_features, segment_characters_from_word, segment_words_from_line
from cnn.cnn import make_prediction_letters, make_prediction_digits


# ------------------------------------------------------
# Visualizations for Streamlit
# ------------------------------------------------------

def viz_original(img):
    """
    DESCRIPTION:
        Creates a Matplotlib figure displaying the raw 28×28 MNIST-style image.

    INPUT:
        img : 2D numpy array

    PROCESSING:
        - Render as grayscale image.
        - Hide axes.

    OUTPUT:
        fig : matplotlib.figure.Figure
            Figure object ready to pass into st.pyplot().
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title("Original Image")
    ax.axis("off")
    return fig


def viz_edges(img):
    """
    DESCRIPTION:
        Displays the Sobel edge map for debugging and visualization.

    INPUT:
        img : 2D numpy array

    PROCESSING:
        - Compute edges via Sobel filter.
        - Plot in grayscale.

    OUTPUT:
        fig : matplotlib.figure.Figure
    """
    edges = compute_edges(img)

    fig, ax = plt.subplots()
    ax.imshow(edges, cmap="gray")
    ax.set_title("Edge Map (Sobel)")
    ax.axis("off")
    return fig


def viz_grid_segments(img, grid_rows=8, grid_cols=8):
    """
    DESCRIPTION:
        Draws a grid overlay used to compute edge-segment features.

    INPUT:
        img : 2D numpy array (edge map)
        grid_rows, grid_cols : ints

    PROCESSING:
        - Overlay grid lines showing segmentation.
        - Display via Matplotlib.

    OUTPUT:
        fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray")
    ax.set_title("Grid Segmentation Overlay")

    H, W = img.shape
    seg_h = H // grid_rows
    seg_w = W // grid_cols

    # horizontal lines
    for r in range(grid_rows + 1):
        ax.axhline(r * seg_h, color="red", linewidth=0.7)

    # vertical lines
    for c in range(grid_cols + 1):
        ax.axvline(c * seg_w, color="red", linewidth=0.7)

    ax.axis("off")
    return fig


def viz_feature_vector(features):
    """
    DESCRIPTION:
        Returns a bar plot showing the edge density per segment.

    INPUT:
        features : 1D numpy array

    PROCESSING:
        - Draw a bar chart.
        - Label axes.

    OUTPUT:
        fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    ax.bar(range(len(features)), features)
    ax.set_title("Feature Vector (Edge Densities)")
    ax.set_xlabel("Segment Index")
    ax.set_ylabel("Density")
    return fig

# ======================
# Utility Functions
# ======================

def preprocess_canvas(img):
    """
    DESCRIPTION:
        Converts the raw RGBA canvas output from Streamlit’s drawing widget
        into a normalized 28×28 grayscale image that matches MNIST formatting.
        This ensures that the custom ANN and the CNN both receive inputs
        in the same standardized structure.

    INPUT:
        img : np.ndarray
            Raw canvas image from Streamlit (H×W×4, uint8 RGBA formatting).

    PROCESSING:
        - Convert RGBA image into grayscale using OpenCV.
        - Resize image to 28×28 pixels to match MNIST dataset.
        - Convert the image to float32 and scale pixel values to [0, 1].

    OUTPUT:
        np.ndarray
            A 28×28 float32 grayscale image normalized to MNIST format.
    """

    # Convert RGBA → grayscale.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to MNIST dimensions.
    img = cv2.resize(img, (28, 28))

    # Normalize pixel intensity.
    img = img.astype("float32") / 255.0

    img = 1 - img 

    return img



def predict_ann_single_digits(img28):
    """
    DESCRIPTION:
        Generates a digit prediction using the custom manually-implemented
        artificial neural network. Internally extracts edge-segment features,
        loads learned weights, and performs forward propagation.

    INPUT:
        img28 : np.ndarray
            A 28×28 MNIST-style grayscale float32 image.

    PROCESSING:
        - Compute the 8×8 grid edge-density features.
        - Flatten the feature vector to shape (1, 64).
        - Load model weights + biases from NPZ file.
        - Print diagnostics (weight means & std) for debugging.
        - Run ANN forward propagation through predict_classes().

    OUTPUT:
        int
            Predicted digit (0–9).
    """

    # Extract grid-based edge feature vector.
    feats = extract_edge_segment_features(img28, grid_rows=8, grid_cols=8)

    # Reshape → single-sample input row.
    feats = feats.reshape(1, -1)

    # Load ANN model parameters.
    weights, biases = load_model("custom_ann_model_digits.npz")

    # Debugging metrics for verifying training health.
    print("W1 mean, std:", np.mean(weights[0]), np.std(weights[0]))
    print("W2 mean, std:", np.mean(weights[1]), np.std(weights[1]))
    print("B1 mean, std:", np.mean(biases[0]), np.std(biases[0]))
    print("B2 mean, std:", np.mean(biases[1]), np.std(biases[1]))

    # Get probabilities from the ANN
    probs = predict_proba(feats, weights, biases)  
    final_activations = probs[0] 

    # Class prediction from probabilities
    pred = int(np.argmax(final_activations))

    return pred, final_activations


def predict_ann_single_letters(img28):
    """
    DESCRIPTION:
        Generates a digit prediction using the custom manually-implemented
        artificial neural network. Internally extracts edge-segment features,
        loads learned weights, and performs forward propagation.

    INPUT:
        img28 : np.ndarray
            A 28×28 MNIST-style grayscale float32 image.

    PROCESSING:
        - Compute the 8×8 grid edge-density features.
        - Flatten the feature vector to shape (1, 64).
        - Load model weights + biases from NPZ file.
        - Print diagnostics (weight means & std) for debugging.
        - Run ANN forward propagation through predict_classes().

    OUTPUT:
        int
            Predicted digit (0–9).
    """

    # Extract grid-based edge feature vector.
    feats = extract_edge_segment_features(img28, grid_rows=8, grid_cols=8)

    # Reshape → single-sample input row.
    feats = feats.reshape(1, -1)

    # Load ANN model parameters.
    weights, biases = load_model("custom_ann_model_letters.npz")

    # Debugging metrics for verifying training health.
    print("W1 mean, std:", np.mean(weights[0]), np.std(weights[0]))
    print("W2 mean, std:", np.mean(weights[1]), np.std(weights[1]))
    print("B1 mean, std:", np.mean(biases[0]), np.std(biases[0]))
    print("B2 mean, std:", np.mean(biases[1]), np.std(biases[1]))

    # Get probabilities from the ANN
    probs = predict_proba(feats, weights, biases)  
    final_activations = probs[0] 

    # Class prediction from probabilities
    pred = int(np.argmax(final_activations))

    return pred, final_activations


@st.cache_resource
def load_tf_digit_cnn():
    return tf.keras.models.load_model("tf_cnn_model_digits.keras", compile=False)

def predict_cnn_single_digits(img28):
    """
    DESCRIPTION:
        Produces a digit prediction using the trained TensorFlow CNN model.
        This serves as the accuracy benchmark against the custom ANN.

    INPUT:
        img28 : np.ndarray
            A 28×28 grayscale image (float32) in MNIST format.

    PROCESSING:
        - Load TensorFlow CNN model from disk.
        - Reshape image → (1, 28, 28, 1) to match Keras expectations.
        - Model predicts probability distribution.
        - Return class with highest probability.

    OUTPUT:
        int
            Predicted digit (0–9).
    """

    # Load CNN model architecture + weights.
    #model = tf.keras.models.load_model("tf_cnn_model.keras")
    model = load_tf_digit_cnn()

    # Reshape → batch format expected by Keras.
    x = img28.reshape(1, 28, 28, 1)

    # Predict class probabilities & take argmax.
    return int(np.argmax(model.predict(x)))

@st.cache_resource
def load_tf_letter_cnn():
    return tf.keras.models.load_model("tf_cnn_model_letters.keras", compile=False)

def predict_cnn_single_letters(img28):
    """
    DESCRIPTION:
        Produces a digit prediction using the trained TensorFlow CNN model.
        This serves as the accuracy benchmark against the custom ANN.

    INPUT:
        img28 : np.ndarray
            A 28×28 grayscale image (float32) in MNIST format.

    PROCESSING:
        - Load TensorFlow CNN model from disk.
        - Reshape image → (1, 28, 28, 1) to match Keras expectations.
        - Model predicts probability distribution.
        - Return class with highest probability.

    OUTPUT:
        int
            Predicted digit (0–9).
    """

    # Load CNN model architecture + weights.
    #model = tf.keras.models.load_model("tf_cnn_model.keras")
    model = load_tf_letter_cnn()

    # Reshape → batch format expected by Keras.
    x = img28.reshape(1, 28, 28, 1)

    # Predict class probabilities & take argmax.
    return int(np.argmax(model.predict(x)))

def idx_to_letter(idx: int) -> str:
    # 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z'
    return chr(ord('A') + int(idx))

# ======================
# Streamlit App Layout
# ======================

# ======================
# DIGITS
# ======================

# App header + description.
st.title("🖊️ Handwritten Digit Recognition Demo")
st.write("Draw a digit below and compare predictions from:")
st.write("- Our custom **ANN model**")
st.write("- Our custom **CNN model**")
st.write("- Standard **TensorFlow CNN**")

# Import canvas tool for drawing.
from streamlit_drawable_canvas import st_canvas

# Create interactive drawing area.
canvas = st_canvas(
    fill_color="rgba(255,255,255,1)",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas_digit"
)

# When a drawing exists on the canvas...
if canvas.image_data is not None:
    # Convert to uint8 image array.
    img = canvas.image_data.astype("uint8")
    print("Begin Raw\n")
    print(img)
    print("\nEnd Raw")

    # Show raw drawing.
    st.image(canvas.image_data, caption="Live Drawing", width=280)

    # Convert to MNIST-style.
    img28 = preprocess_canvas(img)

    # Display preprocessed image for verification.
    #st.subheader("Processed 28×28 Image")
    #st.image(img28, width=150, clamp=True)

    print("RAW IMG SHAPE:", img28.shape)

    # Compute feature vector for debugging.
    feats = extract_edge_segment_features(img28, grid_rows=8, grid_cols=8)

    print("FEATURES:", feats)
    print("FEATURE VECTOR SUM:", np.sum(feats))
    print("FEATURE VECTOR MAX:", np.max(feats))
    print("FEATURE VECTOR MIN:", np.min(feats))

    # -------------------------
    # Compute ANN + CNN predictions
    # -------------------------
    ann_pred, ann_outcomes = predict_ann_single_digits(img28)
    tf_pred = predict_cnn_single_digits(img28)
    cnn_pred, outcomes = make_prediction_digits(img28)
    print(outcomes)
    RED = "\033[31m"
    RESET = "\033[0m"
    for row in img28:
        for num in row:
            #num = 1 - num
            s = f"{num:05.2f}"
            if num != 0.0:
                print(f"{RED}{s}{RESET}", end = ' ')
            else:
                print(s, end = ' ')
        print()
    print("==================================================================")

    # Display side-by-side results.
    st.subheader("Predictions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Custom ANN", ann_pred)
    col2.metric("Custom CNN", cnn_pred)
    col3.metric("TF CNN", tf_pred)

    st.subheader("Original 28x28 Image")
    st.pyplot(viz_original(img28))

    st.subheader("Sobel Edge Map")
    edges = compute_edges(img28)
    st.pyplot(viz_edges(img28))

    st.subheader("Grid Segmentation Overlay")
    st.pyplot(viz_grid_segments(edges, grid_rows=8, grid_cols=8))

    st.subheader("Edge-Segment Feature Vector")
    feats = extract_edge_segment_features(img28, grid_rows=8, grid_cols=8)
    st.pyplot(viz_feature_vector(feats))

    st.subheader("Custom CNN Probabilities")
    st.bar_chart(outcomes)

    st.subheader("Custom ANN Probabilities")
    st.bar_chart(ann_outcomes)

#=============================Letters Begins Here=============================#
st.title("🔡 Handwritten Letters Recognition Demo")

# Create interactive drawing area.
canvas = st_canvas(
    fill_color="rgba(255,255,255,1)",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas_letters"
)

# When a drawing exists on the canvas...
if canvas.image_data is not None:
    # Convert to uint8 image array.
    img = canvas.image_data.astype("uint8")
    print("Begin Raw\n")
    print(img)
    print("\nEnd Raw")

    # Show raw drawing.
    st.image(canvas.image_data, caption="Live Drawing", width=280)

    # Convert to MNIST-style.
    img28 = preprocess_canvas(img)

    # Display preprocessed image for verification.
    st.subheader("Processed 28×28 Image")
    st.image(img28, width=150, clamp=True)

    print("RAW IMG SHAPE:", img28.shape)

    # Compute feature vector for debugging.
    feats = extract_edge_segment_features(img28, grid_rows=8, grid_cols=8)

    print("FEATURES:", feats)
    print("FEATURE VECTOR SUM:", np.sum(feats))
    print("FEATURE VECTOR MAX:", np.max(feats))
    print("FEATURE VECTOR MIN:", np.min(feats))

    # -------------------------
    # Compute ANN + CNN predictions
    # -------------------------
    ann_pred, ann_outcomes = predict_ann_single_letters(img28)
    tf_pred = predict_cnn_single_letters(img28)
    cnn_pred, outcomes = make_prediction_letters(img28)
    print(outcomes)
    RED = "\033[31m"
    RESET = "\033[0m"
    for row in img28:
        for num in row:
            #num = 1 - num
            s = f"{num:05.2f}"
            if num != 0.0:
                print(f"{RED}{s}{RESET}", end = ' ')
            else:
                print(s, end = ' ')
        print()
    print("==================================================================")

    # Display side-by-side results.
    st.subheader("Predictions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Custom ANN", idx_to_letter(ann_pred))
    col2.metric("Custom CNN", idx_to_letter(cnn_pred))
    col3.metric("TF CNN", idx_to_letter(tf_pred))

    st.subheader("Original 28x28 Image")
    st.pyplot(viz_original(img28))

    st.subheader("Sobel Edge Map")
    edges = compute_edges(img28)
    st.pyplot(viz_edges(img28))

    st.subheader("Grid Segmentation Overlay")
    st.pyplot(viz_grid_segments(edges, grid_rows=8, grid_cols=8))

    st.subheader("Edge-Segment Feature Vector")
    feats = extract_edge_segment_features(img28, grid_rows=8, grid_cols=8)
    st.pyplot(viz_feature_vector(feats))

    st.subheader("Custom CNN Probabilities")
    st.bar_chart(outcomes)

    st.subheader("Custom ANN Probabilities")
    st.bar_chart(ann_outcomes)

# =================
# WORDS
# =================

st.title("📝 Handwritten Word → Text with Bounding Boxes")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=128,
    width=512,
    drawing_mode="freedraw",
    key="canvas_word",
)

if st.button("Recognize Word"):
    if canvas_result.image_data is None:
        st.warning("Draw a word first!")
    else:
        img = canvas_result.image_data.astype("uint8")

        # Segment into characters + get bounding boxes
        char_imgs, boxes = segment_characters_from_word(img, return_boxes=True)

        if not char_imgs:
            st.warning("No characters detected. Try writing bigger / darker / more separated.")
        else:
            letters = []

            for char28 in char_imgs:
                # Use your existing CNN letter predictor
                pred_idx, _ = make_prediction_letters(char28)
                letters.append(idx_to_letter(pred_idx))

            word = "".join(letters)
            st.subheader(f"Predicted text: **{word}**")

            # Draw boxes + labels
            if img.shape[2] == 4:
                vis = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                vis = img.copy()

            for (box, letter) in zip(boxes, letters):
                x, y, w, h = box
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(
                    vis, letter,
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    1,
                    lineType=cv2.LINE_AA,
                )

            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            st.image(vis_rgb, caption="Detected letters with bounding boxes", use_container_width=True)

st.title("🧾 Handwritten Sentence → Words")

sentence_canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=128,
    width=800,
    drawing_mode="freedraw",
    key="canvas_sentence",
)

if st.button("Recognize Sentence"):
    if sentence_canvas.image_data is None:
        st.warning("Draw a sentence first!")
    else:
        img = sentence_canvas.image_data.astype("uint8")

        word_char_imgs, word_char_boxes = segment_words_from_line(
            img, return_boxes=True
        )

        if not word_char_imgs:
            st.warning("No characters detected. Try writing bigger / darker.")
        else:
            word_strings = []

            for chars_in_word in word_char_imgs:
                letters = []
                for char28 in chars_in_word:
                    pred_idx, _ = make_prediction_letters(char28)
                    letters.append(idx_to_letter(pred_idx))
                word_strings.append("".join(letters))

            sentence = " ".join(word_strings)
            st.subheader(f"Predicted sentence: **{sentence}**")

            # Visualize word/char boxes on the original image
            if img.shape[2] == 4:
                vis = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                vis = img.copy()

            for boxes, word in zip(word_char_boxes, word_strings):
                for (x, y, w, h) in boxes:
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            st.image(vis_rgb, caption="Segmented words/characters", use_container_width=True)
