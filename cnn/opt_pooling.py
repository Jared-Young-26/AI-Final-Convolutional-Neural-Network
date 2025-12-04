import numpy as np

def maxPool2D_layer(X, dims, stride):
    """
    Performs max pooling on a 2D input array and returns the feature map along with max indices.

    Parameters:
    - X: 2D numpy array, the input data
    - dims: tuple, (pool_height, pool_width) for pooling dimensions
    - stride: tuple, (stride_height, stride_width) for stride dimensions

    Returns:
    - feature_map: 2D numpy array, the result of the max pooling operation
    - indices: list of tuples, containing the relative indices of the max values in each pooling region
    """
    if len(X.shape) != 2 or len(dims) != 2:
        raise ValueError("Input and pool dimensions must be two dimensional")

    pool_height, pool_width = dims
    stride_height, stride_width = stride
    output_height = (X.shape[0] - pool_height) // stride_height + 1
    output_width = (X.shape[1] - pool_width) // stride_width + 1
    
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Pooling size must be less than or equal to input size considering the stride")

    feature_map = np.zeros((output_height, output_width))
    indices = []

    for i in range(output_height):
        for j in range(output_width):
            start_i = i * stride_height
            end_i = start_i + pool_height
            start_j = j * stride_width
            end_j = start_j + pool_width
            
            subarray = X[start_i:end_i, start_j:end_j]
            max_index_subarray = np.argmax(subarray)
            max_index_2d = np.unravel_index(max_index_subarray, subarray.shape)
            max_value = subarray[max_index_2d]
            
            feature_map[i, j] = max_value
            indices.append((max_index_2d[0], max_index_2d[1]))

    return feature_map, indices


def maxPool2D(X, dims, stride):
    """
    Applies max pooling channel-wise to a 3D input array and returns pooled features
    along with indices of max values.

    Parameters:
    - X: 3D numpy array, the input data (channels, height, width)
    - dims: tuple, (pool_height, pool_width) for pooling dimensions
    - stride: tuple, (stride_height, stride_width) for stride dimensions

    Returns:
    - pooled_features: 3D numpy array containing pooled features from all channels
    - indices: list of lists, where each inner list contains the indices of the max values for each channel
    """
    if len(X.shape) != 3 or len(dims) != 2:
        raise ValueError("Input must be 3D and dimensions must be 2D")

    channels = X.shape[0]
    pooled_height = (X.shape[1] - dims[0]) // stride[0] + 1
    pooled_width = (X.shape[2] - dims[1]) // stride[1] + 1

    pooled_features = np.zeros((channels, pooled_height, pooled_width))
    indices = []

    for j in range(channels):
        pooled_features[j], channel_indices = maxPool2D_layer(X[j], dims, stride)
        indices.append(channel_indices)

    return pooled_features, indices


def reverse_max2D_layer(feature_map, local_coords, pool, stride, original_shape):
    """
    Reverses the max pooling operation for a single channel.

    Parameters:
    - feature_map: 2D numpy array, the pooled feature map
    - local_coords: list of tuples, each containing local coordinates of max values
    - pool: tuple, (pool_height, pool_width)
    - stride: tuple, (stride_height, stride_width)
    - original_shape: tuple, (height, width) of the original input channel

    Returns:
    - expanded: 2D numpy array with the same resolution as the original input channel
    """
    # Unpack original dimensions
    original_height, original_width = original_shape

    # Create an array for the expanded output of original size
    expanded = np.zeros((original_height, original_width))

    for i in range(feature_map.shape[0]):
        for j in range(feature_map.shape[1]):
            value = feature_map[i, j]
            local_coord = local_coords[i * feature_map.shape[1] + j]

            # Global coordinates based on local coordinates and stride
            global_i = i * stride[0] + local_coord[0]
            global_j = j * stride[1] + local_coord[1]

            # Ensure we do not go out of bounds
            if global_i < original_height and global_j < original_width:
                expanded[global_i, global_j] = value

    return expanded


def reverse_max2D(pooled_features, indices, stride, dims, original_shape):
    """
    Reverses the max pooling operation channel-wise.

    Parameters:
    - X: 3D numpy array, original input data (channels, height, width)
    - pooled_features: 3D numpy array, pooled feature map
    - indices: list of lists, where each inner list contains the indices of the max values for each channel
    - stride: tuple, (stride_height, stride_width)
    - dims: tuple, (pool_height, pool_width)

    Returns:
    - expanded_features: 3D numpy array with expanded features matching the original input dimensions
    """
    channels = original_shape[0]
    expanded_features = np.zeros(original_shape)

    for j in range(channels):
        # Pass original shape to handle expansion correctly
        expanded_features[j] = reverse_max2D_layer(pooled_features[j], indices[j], dims, stride, original_shape[1:])

    return expanded_features

