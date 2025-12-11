from tensorflow.keras import datasets
import tensorflow_datasets as tfds
from random import randint
from emnist import extract_training_samples
import numpy as np

from cnn.cnn import run_model, make_prediction

import numpy as np
from cnn.init import sobel

def run():
    print("==Loading Data MNIST Digits==")
    
    if False:
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data();
    else:
        dataset, info = tfds.load('emnist/letters', with_info=True, as_supervised=True)
        train, test = dataset["train"], dataset["test"]
        x_train, y_train = [], []
        x_test, y_test = [], []

        # Collect data
        for image, label in train:
            x_train.append(image)
            y_train.append(label)

        for image, label in test:
            x_test.append(image)
            y_test.append(label)

        # Convert lists to NumPy arrays
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        x_train = np.squeeze(x_train)
        y_train = np.squeeze(y_train)
        x_test = np.squeeze(x_test)
        y_test = np.squeeze(y_test)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print(np.min(y_train), np.max(y_train))

    y_train -= 1
    y_test -= 1

    TRAINING_SAMPLES = x_train.shape[0]
    EPOCHS = 5
    LEARNING_RATE = 0.005
    
    #Normalize Data
    x_train = x_train / 255
    x_test = x_test / 255

    print("==Data Loaded==\n")

    res = make_prediction(x_test[0])
    print(res, y_test[0])
    
    #run_model(
    #    x_train, y_train, x_test, y_test, 
    #    EPOCHS, LEARNING_RATE, TRAINING_SAMPLES
    #) 

    print("\nFinished")
    
if __name__ == "__main__": 
    #data = np.load("custom_model.npz")
    #print(data)
    #new_data = {**data, 'kernel': sobel()}
    #np.savez("custom_model.npz", **new_data)
    run()
