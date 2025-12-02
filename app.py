import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from ann import load_model, predict_classes
from extra import compute_edges, extract_edge_segment_features, segment_edge_image


# ======================
# Utility Functions
# ======================

def preprocess_canvas(img):
    """
    Convert canvas image -> 28x28 MNIST style float32 array.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0

    # invert: canvas draws black on white; MNIST is white on black
    img = 1.0 - img
    return img


def predict_ann_single(img28):
    feats = extract_edge_segment_features(img28)
    feats = feats.reshape(1, -1)

    weights, biases = load_model("custom_ann_model.npz")
    pred = predict_classes(feats, weights, biases)[0]
    return pred


def predict_cnn_single(img28):
    model = tf.keras.models.load_model("tf_cnn_model.keras")
    x = img28.reshape(1, 28, 28, 1)
    return int(np.argmax(model.predict(x)))


# ======================
# Streamlit App Layout
# ======================

st.title("🖊️ Handwritten Digit Recognition Demo")
st.write("Draw a digit below and compare predictions from:")
st.write("- Your custom **ANN model**")
st.write("- Standard **TensorFlow CNN**")

# Canvas component
from streamlit_drawable_canvas import st_canvas

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

if canvas.image_data is not None:
    img = canvas.image_data.astype("uint8")
    st.image(canvas.image_data, caption="Live Drawing", width=280)
    img28 = preprocess_canvas(img)

    # Display 28x28 processed image
    st.subheader("Processed 28×28 Image")
    st.image(img28, width=150, clamp=True)

    # Predictions
    ann_pred = predict_ann_single(img28)
    cnn_pred = predict_cnn_single(img28)

    st.subheader("Predictions")
    col1, col2 = st.columns(2)
    col1.metric("Custom ANN", ann_pred)
    col2.metric("TF CNN", cnn_pred)

    # Edge Visualization
    st.subheader("Sobel Edge Map")
    edges = compute_edges(img28)
    st.image(edges, width=150, clamp=True)

    # Feature Vector
    st.subheader("Edge-Segment Feature Vector")
    feats = extract_edge_segment_features(img28)
    fig, ax = plt.subplots()
    ax.bar(range(len(feats)), feats)
    st.pyplot(fig)

