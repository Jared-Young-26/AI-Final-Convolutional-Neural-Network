import numpy as np

def conv2D_layer(X, kernel):
    if len(X.shape) != 2 or len(kernel.shape) != 2:
        raise ValueError("Input and kernel must be two dimensional")

    # Use numpy's array manipulation for convolution
    feature_map = np.zeros(
        (X.shape[0] - kernel.shape[0] + 1, X.shape[1] - kernel.shape[1] + 1)
    )

    # Using np.einsum for optimized element-wise multiplication and summation
    for i in range(feature_map.shape[0]):
        for j in range(feature_map.shape[1]):
            feature_map[i, j] = np.sum(X[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)
    
    return feature_map


def conv2D(X, kernel):
    if len(X.shape) != 3 or len(kernel.shape) != 4:
        raise ValueError("Input must be 3D and kernel must be 4D")

    channel_out = kernel.shape[0]
    feature_maps = np.zeros((channel_out, X.shape[1] - kernel.shape[2] + 1, X.shape[2] - kernel.shape[3] + 1))

    for i in range(channel_out):
        for j in range(X.shape[0]):  # Iterate over input channels
            feature_maps[i] += conv2D_layer(X[j], kernel[i][j])

    return feature_maps
    
def main(debug = False):
    channel_in = 1
    channel_out = 32
    image_height = 28
    image_width = 28
    kernel_height = 5
    kernel_width = 5

    if not debug:
        X = np.random.rand(channel_in, image_height, image_width)
        kernel = np.random.rand(channel_out, channel_in, kernel_height, kernel_width)
    else:
        X = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]) 
        kernel = np.array([[[[0, 1], [2, 3]], [[1, 2], [3, 4]]]])    
     
    L1 = conv2D(X, kernel)
    print(L1)
    print("Finished")

if __name__ == "__main__":
    main(debug = True) 
