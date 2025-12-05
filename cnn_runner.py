from tensorflow.keras import datasets
from random import randint

from cnn.cnn import run_model, make_prediction

def run(train=False, tests=20):
    print("==Loading Data==")
    
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data();
    x_train = x_train / 255
    x_test = x_test / 255

    print("==Data Loaded==\n")
    
    if train:
        run_model(x_train, y_train, x_test, y_test)
    
    for _ in range(tests):
        x = randint(0, 10000)
        print(f"Model Prediction: {make_prediction(x_test[x])} ", end='')
        print(f"Actual: {y_test[x]}")
    
if __name__ == "__main__":
    run()
