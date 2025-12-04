import numpy as np

def one_hot_encode(index, vector_length):
    one_hot_vector = np.zeros(vector_length)
    one_hot_vector[index] = 1
    return one_hot_vector
