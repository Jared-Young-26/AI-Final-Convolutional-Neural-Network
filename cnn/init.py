import numpy as np

def he_init(out_dim, in_dim):
    # Calculate standard deviation
    stddev = np.sqrt(2. / in_dim)
    # Generate weights from a normal distribution
    weights = np.random.normal(loc=0.0, scale=stddev, size=(out_dim, in_dim))
    return weights

def xavier_init(out_dim, in_dim):
    # Calculate the limit for uniform distribution
    limit = np.sqrt(6 / (in_dim + out_dim))
    weights = np.random.uniform(-limit, limit, (out_dim, in_dim))
    return weights

def sobel():
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    sobel = np.zeros((2, 1, 3, 3), dtype=np.float32)

    sobel[0:, :, :, :] = sobel_x
    sobel[1:, :, :, :] = sobel_y

    return sobel

