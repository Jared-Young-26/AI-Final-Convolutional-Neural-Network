from tensorflow.keras import datasets

from cnn import run_model

if __name__ == "__main__":
    print("==Loading Data==")
    
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data();
    x_train = x_train / 255
    x_test = x_test / 255

    print("==Data Loaded==\n")

    run_model(x_train, y_train, x_test, y_test)
    
