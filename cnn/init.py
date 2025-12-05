import numpy as np

def he_init(out_dim, in_dim):
    # Calculate standard deviation
    stddev = np.sqrt(2. / in_dim)
    # Generate weights from a normal distribution
    weights = np.random.normal(loc=0.0, scale=stddev, size=(out_dim, in_dim))
    return weights

def he_init_conv(out_channels, in_channels, kernel_height, kernel_width):
    # Calculate the fan-in
    fan_in = kernel_height * kernel_width * in_channels
    # Calculate standard deviation
    stddev = np.sqrt(2. / fan_in)
    # Generate kernels from a normal distribution
    kernels = np.random.normal(loc=0.0, scale=stddev, size=(out_channels, in_channels, kernel_height, kernel_width))
    return kernels
