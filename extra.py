import numpy as np

def compute_edges(image, threshold=0.2):
    """
    Compute a simple edge map using Sobel-like filters (no external libs).
    image: 2D array (H, W) with values in [0, 1] or [0, 255].
    Returns: binary edge map (H, W) with values 0 or 1.
    """
    # Normalize if needed
    if image.max() > 1.5:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)
    
    # Simple Sobel kernels for horizontal and vertical gradients
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    
    H, W = img.shape
    Gx = np.zeros_like(img)
    Gy = np.zeros_like(img)
    
    # Convolution (valid region only, ignore 1-pixel border)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            region = img[i-1:i+2, j-1:j+2]
            Gx[i, j] = np.sum(region * Kx)
            Gy[i, j] = np.sum(region * Ky)
    
    # Gradient magnitude
    mag = np.sqrt(Gx**2 + Gy**2)
    mag = mag / (mag.max() + 1e-8)  # normalize to [0,1]
    
    # Threshold to binary edge map
    edges = (mag > threshold).astype(np.float32)
    return edges


def segment_edge_image(edge_img, grid_rows=4, grid_cols=4):
    """
    Divide the edge image into grid_rows x grid_cols segments and compute
    edge density in each segment.
    edge_img: 2D binary array (H, W), values 0 or 1.
    Returns: 1D feature vector of length (grid_rows * grid_cols).
    """
    H, W = edge_img.shape
    seg_h = H // grid_rows
    seg_w = W // grid_cols
    
    features = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            r_start = r * seg_h
            c_start = c * seg_w
            # last segments may be slightly larger if H/W not divisible; handle with min()
            r_end = H if r == grid_rows - 1 else (r_start + seg_h)
            c_end = W if c == grid_cols - 1 else (c_start + seg_w)
            
            segment = edge_img[r_start:r_end, c_start:c_end]
            # Edge density = fraction of pixels that are edges
            density = segment.mean()  # since edges are 0/1
            features.append(density)
    
    return np.array(features, dtype=np.float32)


def extract_edge_segment_features(image, grid_rows=4, grid_cols=4, threshold=0.2):
    """
    Combined pipeline: image -> edge map -> grid segmentation -> feature vector.
    image: 2D numpy array (H,W).
    Returns: 1D feature vector (grid_rows * grid_cols).
    """
    edges = compute_edges(image, threshold=threshold)
    features = segment_edge_image(edges, grid_rows, grid_cols)
    return features
