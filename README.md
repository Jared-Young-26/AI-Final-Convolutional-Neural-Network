# AI-Final-Convolutional-Neural-Network

Write a neural network program to recognize the handwritten digits / alphabets by dividing
the characters as a set of edge segments and inputting the segments for classification

## Motivation

While convolutional neural networks are relativly well understood at a high level, and much of the research is in low level implementation details such as hardware or underlying algorithms generally abstracted away from users, as when most people think about convolutional or neural networks in general, thoughts usally go to PyTorch or Tensorflow or other such libraries. However, there is still much insight to be gained to be gained from understanding how they work at a higher level and trying to implement them without these higher level libraries as to really understand how they work it helps to delve right in the deep end. Aside from that, to understand why convolultional neural networks are used, it would be helpful to compare them to a traditional neural networks to compare the pros on cons of each one and see difference results each produces. This way metrics like accuracy, and resources like time and computational power can be compared.

The goal of this project was to reverse that dynamic:
build a neural network almost entirely from scratch, train it on MNIST using hand-designed edge-segment features, and compare its behavior, limitations, and performance to a standard convolutional neural network (CNN).

This forced a hands-on engagement with every part of the learning pipeline:
 * feature engineering
 * activation stability
 * weight initialization
 * softmax + cross-entropy math
 * backpropagation
 * debugging exploding/vanishing gradients
 * tensor shape mismatches
 * mini-batch training
 * normalization and scaling
 * architectural choices and hyperparameters
 * model serialization

A TensorFlow CNN was also implemented—not as the “main model,” but as a scientific baseline to show how much automatic optimization modern frameworks handle under the hood.

## Project Description

This project implements two complete classification pipelines:

### 1. Custom Manual Neural Network (ANN)
A feed-forward neural network implemented entirely using NumPy, including:
 * Custom activation functions (ReLU, tanh, sigmoid)
 * Manually implemented backpropagation
 * Softmax output with cross-entropy loss
 * Mini-batch gradient descent
 * Standardization of inputs for stable training
 * Custom weight initialization (Xavier/Glorot uniform)
 * Saving + loading model parameters (np.savez)
 * Debugging instrumentation (weight means, stds, activations)
 * Full documentation of every function and line-by-line explanation

This ANN is intentionally feature-dependent: it does not receive raw pixel values.
Instead, it uses a handcrafted feature extraction pipeline.

### 2. Edge-Segment Feature Extraction

Because the assignment required classification based on edge-segments, the project includes a custom feature engineering module that:
 1. Computes Sobel edges on a 28×28 MNIST digit
 2. Divides the image into an N×N grid (e.g., 8×8 → 64 segments)
 3. Calculates edge density inside each segment
 4. Produces a fixed-length feature vector (16, 32, 64 dims depending on grid size)

These extracted features become the ANN’s input vector.

This allowed experimentation with:
 * different grid resolutions
 * thresholding strategies
 * impact of sparse vs dense feature vectors
 * failure modes when features collapse to zero

### 3. TensorFlow CNN (Comparison Model)

A compact but high-quality CNN was built using TensorFlow/Keras to:
 * benchmark accuracy
 * show the difference between learned feature extraction vs manually engineered ones
 * demonstrate modern training conveniences (dropout, ReLU stacks, pooling)

As expected, the CNN reaches ~98.8–99% accuracy on MNIST, easily outperforming the ANN—
but the project’s goal wasn't to “beat” TensorFlow, but to understand why.

### 4. Streamlit Interactive Application

A fully functional front-end was created using Streamlit:
 * Users draw a digit on a live canvas
 * The drawing is preprocessed → 28×28 grayscale
 * Sobel edges and grid segmentation are computed live
 * Feature vector is displayed as a bar plot
 * Predictions from both models (ANN and CNN) are shown side-by-side
 * Visualizations mirror the analysis done in master.py

This turns the entire project into an interactive, visually intuitive exploration tool.

## Project Design

The project is divided into modular components:

 * ann.py
    Fully documented manual neural network implementation.

 * segments.py / extra.py
    Edge detection, grid segmentation, feature extraction.

 * master.py
    Full training pipeline, dataset loading, ANN training, CNN training, visualization utilities.

 * app.py
    Streamlit interface for interactive digit recognition.

 * Saved model artifacts
    - custom_ann_model.npz
    - tf_cnn_model.keras

The modular design allowed rapid experimentation across:
 * activations (sigmoid → tanh → ReLU)
 * feature vector sizes (16 → 32 → 64)
 * learning rate schedules
 * normalization approaches
 * standardization vs raw features
 * batch sizes
 * random initialization stability

## Struggles

1. Figuring out how to apply multiple convolutional filters to change the number of channels

2. How to apply backpropagation to update the learnable filters

3. Feature vectors collapsing to zero

Many drawn digits produced feature vectors like [0, 0, 0, ...], crippling the ANN.
This led to:
 * modifying Sobel thresholds
 * tuning grid sizes
 * inspecting segment densities through visualization
 * discovering that CNNs bypass this issue because they learn features end-to-end

4. Activation function problems

Sigmoid saturated immediately.
Tanh performed better but still compressed gradients.
ReLU improved training but produced dead neurons unless inputs were standardized.

This directly reinforced the design principles behind CNNs.

5. Gradient instability

Without:
 * standardization
 * Xavier initialization
 * mini-batch training

…the network diverged or flattened.
Correcting these manually produced a working ANN with ~30% accuracy—
a massive improvement over the initial ~9%.

6. Understanding TensorFlow magic

Seeing the contrast between the manual ANN (30% accuracy with engineered features)
and the CNN (~99% accuracy with raw pixels) highlighted:
 * why learned convolutional filters outperform hand-designed features
 * how deep architectures learn abstractions
 * the power of optimized backprop and weight initialization
 * how regularization (dropout, pooling) stabilizes learning

## Results

| Model                  | Input Type                       | Train Accuracy | Test Accuracy  |
| ---------------------- | -------------------------------- | -------------- | -------------- |
| **Manual ANN (NumPy)** | 64-segment edge density features | ~30%           | ~29–30%        |
| **TensorFlow CNN**     | raw pixels                       | ~97–98% train  | ~98.8–99% test |


## Questions or Discussion for Future Work

 * Implement learnable convolution filters by hand (true manual CNN)

 * Add momentum / Adam optimizer to the ANN

 * Experiment with alternative feature engineering (HOG, Canny edges, Gabor filters)

 * Support letters (A–Z) using EMNIST dataset

 * Expand final project paper with visual explanations from Streamlit

 * Add confusion matrices and saliency maps for interpretability

 * Try deeper ANNs with wider feature sets to observe diminishing returns

## References

1. https://en.wikipedia.org/wiki/Convolutional_neural_network

2. https://medium.com/advanced-deep-learning/cnn-operation-with-2-kernels-resulting-in-2-feature-mapsunderstanding-the-convolutional-filter-c4aad26cf32

3. https://www.geeksforgeeks.org/computer-vision/backpropagation-in-convolutional-neural-networks/
