import numpy as np

def maxPool2D_layer(X, dims):
    """
    X is a 2D feature map
    dims is a tuple (x, y) for height of width of pooling region
    Returns 2D max feature map
    Used as a helper function for maxPool2D
    """
    if len(X.shape) != 2 or len(dims) != 2:
        raise Exception("Input and pool must be two dimensional")

    X_height = X.shape[0]
    X_width = X.shape[1]
    pool_height = dims[0]
    pool_width = dims[1]

    map_shape = (X_height - pool_height + 1, X_width - pool_width + 1)
    map_height = map_shape[0]
    map_width = map_shape[1]

    # Maybe just ignore out of bounds results? For future thought
    if map_height < 0 or map_width < 0:
        raise Exception("Pool size must be less or equal to input size")

    feature_map = np.zeros(map_shape)
    for i in range(map_height):
        for j in range(map_width):
            val = max(
                [
                    X[i + x, j + y]
                    for x in range(pool_height) 
                    for y in range(pool_width)
                ]
            )  
            feature_map[i, j] = val
    return feature_map


def maxPool2D(X, dims):
    """
    This function performs the actual pooling on a 3D input X
    """
    if len(X.shape) != 3 or len(dims) != 2:
        raise Exception("Input must be 3D and dims must be 2D")

    channels = X.shape[0]
    X_height = X.shape[1]
    X_width = X.shape[2]

    arrs = [
        maxPool2D_layer(X[j], dims) 
        for j in range(channels)
    ]

    return np.array(arrs)

