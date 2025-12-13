import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from ann import load_model, predict_classes, predict_proba
from segments import compute_edges, extract_edge_segment_features, segment_characters_from_word, segment_words_from_line, preprocess_canvas_to_mnist
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
        Generates a letter prediction using the custom manually-implemented
        artificial neural network. Internally extracts edge-segment features,
        loads learned weights, and performs forward propagation.

    INPUT:
        img28 : np.ndarray
            A 28×28 EMNIST-style grayscale float32 image.

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
        Produces a letter prediction using the trained TensorFlow CNN model.
        This serves as the accuracy benchmark against the custom ANN.

    INPUT:
        img28 : np.ndarray
            A 28×28 grayscale image (float32) in EMNIST format.

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
    stroke_width=6,
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
    img28 = preprocess_canvas_to_mnist(img)
    st.image(img28, caption="Preprocessed MNIST Input", width=150)


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

# ===============
# LETTERS
# ===============

st.title("🔡 Handwritten Letters Recognition Demo")

# Create interactive drawing area.
canvas = st_canvas(
    fill_color="rgba(255,255,255,1)",
    stroke_width=7,
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
    img28 = preprocess_canvas_to_mnist(img)
    st.image(img28, caption="Preprocessed MNIST Input", width=150)

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
    stroke_width=7,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=128,
    width=512,
    drawing_mode="freedraw",
    key="canvas_word",
)

if st.button("Recognize Word"):

    # Guard clause: ensure the user has drawn something on the canvas
    if canvas_result.image_data is None:
        st.warning("Draw a word first!")
    else:
        # Convert Streamlit canvas output --> uint8 image for OpenCV
        img = canvas_result.image_data.astype("uint8")

        # --------------------------------------------------
        # Segment the drawn word into individual characters
        # Returns:
        #   char_imgs -> list of 28x28 normalized character images
        #   boxes     -> corresponding (x, y, w, h) bounding boxes
        # --------------------------------------------------
        char_imgs, boxes = segment_characters_from_word(
            img, return_boxes=True
        )

        # If no characters were found, give user feedback
        if not char_imgs:
            st.warning(
                "No characters detected. Try writing bigger / darker / more separated."
            )
        else:
            letters = []

            # --------------------------------------------------
            # Predict each character independently using CNN
            # --------------------------------------------------
            for char28 in char_imgs:
                # Each char28 is already preprocessed to match training:
                # - centered
                # - square
                # - resized to 28x28
                # - normalized to [0,1]
                pred_idx, _ = make_prediction_letters(char28)

                # Convert numeric class index to actual letter
                letters.append(idx_to_letter(pred_idx))

            # Combine predicted letters into a single word
            word = "".join(letters)
            st.subheader(f"Predicted text: **{word}**")

            # --------------------------------------------------
            # Visualization: draw bounding boxes and predictions
            # --------------------------------------------------
            if img.shape[2] == 4:
                # Convert RGBA → BGR for OpenCV drawing
                vis = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                vis = img.copy()

            # Draw each character box with its predicted label
            for (box, letter) in zip(boxes, letters):
                x, y, w, h = box

                # Bounding box around character
                cv2.rectangle(
                    vis,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    1
                )

                # Draw predicted letter above the box
                cv2.putText(
                    vis,
                    letter,
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    1,
                    lineType=cv2.LINE_AA,
                )

            # Convert back to RGB for Streamlit display
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            st.image(
                vis_rgb,
                caption="Detected letters with bounding boxes",
                use_container_width=True
            )


# =================
# SENTENCES
# =================

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

    # Guard clause: make sure the user actually drew something
    if sentence_canvas.image_data is None:
        st.warning("Draw a sentence first!")
    else:
        # Convert canvas RGBA float image → uint8 for OpenCV processing
        img = sentence_canvas.image_data.astype("uint8")

        # --------------------------------------------------
        # Segment the line into words, then characters
        # Returns:
        #   word_char_imgs  -> [[char28, char28, ...], ...]
        #   word_char_boxes -> matching bounding boxes
        # --------------------------------------------------
        word_char_imgs, word_char_boxes = segment_words_from_line(
            img, return_boxes=True
        )

        # If no valid contours were found, give user feedback
        if not word_char_imgs:
            st.warning("No characters detected. Try writing bigger / darker.")
        else:
            word_strings = []

            # --------------------------------------------------
            # Predict letters word-by-word, character-by-character
            # --------------------------------------------------
            for chars_in_word in word_char_imgs:
                letters = []

                for char28 in chars_in_word:
                    # Each char28 is already normalized and 28×28
                    # CNN predicts class index for a single letter
                    pred_idx, _ = make_prediction_letters(char28)

                    # Convert numeric class index → actual letter
                    letters.append(idx_to_letter(pred_idx))

                # Join predicted letters into a word
                word_strings.append("".join(letters))

            # Join all predicted words into a sentence
            sentence = " ".join(word_strings)
            st.subheader(f"Predicted sentence: **{sentence}**")

            # --------------------------------------------------
            # Visualization: draw character bounding boxes
            # on the original canvas image for debugging
            # --------------------------------------------------
            if img.shape[2] == 4:
                # Convert RGBA to BGR for OpenCV drawing
                vis = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                vis = img.copy()

            # Draw green boxes around each detected character
            for boxes, word in zip(word_char_boxes, word_strings):
                for (x, y, w, h) in boxes:
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

            # Convert back to RGB for Streamlit display
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            st.image(
                vis_rgb,
                caption="Segmented words/characters",
                use_container_width=True
            )
