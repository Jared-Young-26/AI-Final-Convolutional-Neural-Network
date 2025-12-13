import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from ann import load_model, predict_classes, predict_proba
from segments import compute_edges, extract_edge_segment_features, segment_characters_from_word, segment_words_from_line, preprocess_canvas_to_mnist
from cnn.cnn import make_prediction_letters, make_prediction_digits


st.sidebar.title("Mode Selection")

mode = st.sidebar.radio(
    "Choose recognition mode:",
    (
        "Digit",
        "Letter",
        "Word",
        "Sentence",
    )
)

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


def predict_ann(img28, model_path):
    feats = extract_edge_segment_features(img28, grid_rows=8, grid_cols=8)
    feats = feats.reshape(1, -1)

    weights, biases = load_model(model_path)
    probs = predict_proba(feats, weights, biases)[0]

    return int(np.argmax(probs)), probs

@st.cache_resource
def load_tf_model(path):
    return tf.keras.models.load_model(path, compile=False)

def predict_tf(img28, model):
    x = img28.reshape(1, 28, 28, 1)
    probs = model.predict(x, verbose=0)[0]
    return int(np.argmax(probs)), probs

def idx_to_letter(idx: int) -> str:
    return chr(ord("A") + int(idx))

def canvas_to_img28(canvas_img, mode="digit"):
    img28 = preprocess_canvas_to_mnist(canvas_img, mode=mode)
    return img28

# ======================
# Streamlit App Layout
# ======================

# ======================
# DIGITS
# ======================
def run_digit_mode():
    st.title("🖊️ Handwritten Digit Recognition")

    canvas = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=6,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas_digit",
    )

    if canvas.image_data is not None:
        img = canvas.image_data.astype("uint8")
        img28 = canvas_to_img28(img)

        st.image(img28, caption="Preprocessed MNIST Input", width=150)

        ann_pred, ann_probs = predict_ann(img28, "custom_ann_model_digits.npz")
        cnn_pred, cnn_probs = make_prediction_digits(img28)
        tf_pred, tf_probs = predict_tf(img28, load_tf_model("tf_cnn_model_digits.keras"))

        col1, col2, col3 = st.columns(3)
        col1.metric("Custom ANN", ann_pred)
        col2.metric("Custom CNN", cnn_pred)
        col3.metric("TF CNN", tf_pred)

        st.pyplot(viz_original(img28))
        st.pyplot(viz_edges(img28))
        st.pyplot(viz_grid_segments(compute_edges(img28)))
        st.pyplot(viz_feature_vector(extract_edge_segment_features(img28)))

        st.subheader("CNN Probabilities")
        st.bar_chart(cnn_probs)

        st.subheader("ANN Probabilities")
        st.bar_chart(ann_probs)

# ===============
# LETTERS
# ===============
def run_letter_mode():
    st.title("🔡 Handwritten Letter Recognition")

    canvas = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=7,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas_letter",
    )

    if canvas.image_data is not None:
        img = canvas.image_data.astype("uint8")
        img28 = canvas_to_img28(img, mode="letter")

        st.image(img28, caption="Preprocessed EMNIST Input", width=150)

        ann_pred, ann_probs = predict_ann(img28, "custom_ann_model_letters.npz")
        cnn_pred, cnn_probs = make_prediction_letters(img28)
        tf_pred, tf_probs = predict_tf(img28, load_tf_model("tf_cnn_model_letters.keras"))

        col1, col2, col3 = st.columns(3)
        col1.metric("Custom ANN", idx_to_letter(ann_pred))
        col2.metric("Custom CNN", idx_to_letter(cnn_pred))
        col3.metric("TF CNN", idx_to_letter(tf_pred))

        st.pyplot(viz_original(img28))
        st.pyplot(viz_edges(img28))
        st.pyplot(viz_grid_segments(compute_edges(img28)))
        st.pyplot(viz_feature_vector(extract_edge_segment_features(img28)))

        st.subheader("CNN Probabilities")
        st.bar_chart(cnn_probs)

        st.subheader("ANN Probabilities")
        st.bar_chart(ann_probs)


# =================
# WORDS
# =================
def run_word_mode():
    st.title("📝 Handwritten Word Recognition")

    canvas_word = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=7,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=128,
        width=512,
        drawing_mode="freedraw",
        key="canvas_word",
    )

    if st.button("Recognize Word"):
        if canvas_word.image_data is None:
            st.warning("Draw a word first!")
        else:
            img = canvas_word.image_data.astype("uint8")
            char_imgs, boxes = segment_characters_from_word(img, return_boxes=True)

            letters = []
            for char_img in char_imgs:
                # Preprocess each character EXACTLY like single-letter canvas
                char28 = preprocess_canvas_to_mnist(
                    char_img,
                    mode="letter",
                    input_type="canvas"
                )
                #char28 = char_img
                pred_idx, _ = predict_tf(char28, load_tf_model("tf_cnn_model_letters.keras"))
                #pred_idx, _ = make_prediction_letters(char28)
                letters.append(idx_to_letter(pred_idx))


            st.subheader(f"Predicted word: **{''.join(letters)}**")

            vis = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            for (x, y, w, h), letter in zip(boxes, letters):
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 1)
                cv2.putText(vis, letter, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1)

            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)



# =================
# SENTENCES
# =================
def run_sentence_mode():
    st.title("🧾 Handwritten Sentence → Words")

    sentence_canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=7,
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

                    for char_img in chars_in_word:
                        # Apply the SAME preprocessing used everywhere else
                        char28 = preprocess_canvas_to_mnist(
                            char_img,
                            mode="letter",
                            input_type="canvas"
                        )

                        pred_idx, _ = make_prediction_letters(char28)
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

if mode == "Digit":
    run_digit_mode()
elif mode == "Letter":
    run_letter_mode()
elif mode == "Word":
    run_word_mode()
elif mode == "Sentence":
    run_sentence_mode()
