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
    else:
        print("No Traing ... Skipping to Predictions\n")

    print(f"Making {tests} Predictions\n")
   
    for _ in range(tests):
        x = randint(0, 10000)
        pred, outcome = make_prediction(x_test[x])
        act = y_test[x]
        if pred == act:
            print(f"\033[92mModel Prediction: {pred}\033[0m ", end='')
            print(f"\033[92mActual: {y_test[x]}\n\033[0m", end='')
        else:
            print(f"\033[91mModel Prediction: {pred} \033[0m", end='')
            print(f"\033[91mActual: {y_test[x]}\n\033[0m", end='')
        #print(f"\nOutcomes: {outcome}\n\n")

    print("\nFinished")
    
if __name__ == "__main__":
    run()
