import cv2
import numpy as np

def compute_edges(image, threshold=0.001):
    """
    DESCRIPTION:
        Produces a binary edge map from a grayscale image using simplified
        Sobel-like gradient filters. This function exists to convert raw pixel
        intensity data into a structural representation—edges—which often carry
        more meaningful shape information for classification tasks.

    INPUT:
        image (np.ndarray):
            2D grayscale array of shape (H, W). Values may be in [0, 1] or [0, 255].
        threshold (float):
            Normalized cutoff in [0,1] determining what gradient strengths count as edges.

    PROCESSING:
        - Normalize the image to the range [0, 1] if necessary.
        - Apply horizontal and vertical Sobel filters to estimate intensity gradients.
        - Compute gradient magnitude at each pixel.
        - Normalize gradient magnitude so values fall in [0, 1].
        - Threshold the normalized gradient to produce a binary edge map.

    OUTPUT:
        edges (np.ndarray):
            Binary 2D array (H, W) of dtype float32, where:
                1.0 = edge pixel
                0.0 = non-edge pixel
    """
    # --- Normalize image to [0, 1] ---
    # If pixel values exceed 1.5, assume 0–255 and scale down.
    if image.max() > 1.5:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    # --- Define Sobel-like horizontal (Kx) and vertical (Ky) gradient filters ---
    # These filters approximate local intensity changes in x and y directions.
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    H, W = img.shape

    # Gradient containers for horizontal and vertical responses
    Gx = np.zeros_like(img)
    Gy = np.zeros_like(img)

    # --- Convolution: manually slide 3×3 Sobel kernels across the image ---
    # Ignore the 1-pixel border because edges cannot be computed there.
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            region = img[i-1:i+2, j-1:j+2]  # 3×3 region centered at (i, j)
            Gx[i, j] = np.sum(region * Kx)  # horizontal gradient strength
            Gy[i, j] = np.sum(region * Ky)  # vertical gradient strength

    # --- Gradient magnitude ---
    # Combines Gx and Gy into a single estimate of edge strength.
    mag = np.sqrt(Gx**2 + Gy**2)

    # Normalize gradient magnitude so thresholding is consistent across images
    mag = mag / (mag.max() + 1e-8)

    # --- Binary thresholding ---
    # Pixels with strong gradients are labeled as edges.
    edges = (mag > threshold).astype(np.float32)

    return edges


def segment_edge_image(edge_img, grid_rows=8, grid_cols=8):
    """
    DESCRIPTION:
        Converts a binary edge map into a structured feature vector by dividing
        the image into a uniform grid and measuring edge density in each cell.
        This provides the ANN with coarse structural information about where
        edges occur in the image.

    INPUT:
        edge_img (np.ndarray):
            2D binary array from compute_edges().
        grid_rows (int):
            Number of vertical segments.
        grid_cols (int):
            Number of horizontal segments.

    PROCESSING:
        - Divide the image into grid_rows × grid_cols segments.
        - For each segment, calculate edge density (mean of 0/1 values).
        - Flatten all segment densities into a 1D feature vector.

    OUTPUT:
        features (np.ndarray):
            1D float32 vector of length grid_rows * grid_cols.
            Each value is the edge density for one segment.
    """
    H, W = edge_img.shape
    seg_h = H // grid_rows
    seg_w = W // grid_cols

    features = []

    # --- Iterate across each grid cell ---
    for r in range(grid_rows):
        for c in range(grid_cols):

            # Compute segment boundaries (handle last row/col carefully)
            r_start = r * seg_h
            c_start = c * seg_w
            r_end = H if r == grid_rows - 1 else r_start + seg_h
            c_end = W if c == grid_cols - 1 else c_start + seg_w

            # Extract segment of the edge map
            segment = edge_img[r_start:r_end, c_start:c_end]

            # Compute density = proportion of pixels that are edges
            density = segment.mean()

            features.append(density)

    return np.array(features, dtype=np.float32)


def extract_edge_segment_features(image, grid_rows=8, grid_cols=8, threshold=0.001):
    """
    DESCRIPTION:
        Full pipeline step combining edge detection and grid segmentation.
        Converts an input image into a fixed-length structural feature vector.
        This function exists as the pre-processing stage for our custom ANN.

    INPUT:
        image (np.ndarray):
            2D grayscale image of shape (H, W).
        grid_rows, grid_cols (int):
            Specifies segmentation granularity.
        threshold (float):
            Edge threshold to use in compute_edges().

    PROCESSING:
        - Run edge detection using compute_edges().
        - Segment the resulting edge map with segment_edge_image().
        - Return the flattened density vector.

    OUTPUT:
        features (np.ndarray):
            1D float32 array of length grid_rows * grid_cols.
    """
    # Compute binary edge map
    edges = compute_edges(image, threshold=threshold)

    # Convert edge map into structural density features
    features = segment_edge_image(edges, grid_rows, grid_cols)

    return features

def segment_characters_from_word(img, min_area=20, dilate=True, return_boxes=False):
    """
    Given a canvas image with multiple handwritten characters,
    return a list of 28x28 normalized character images,
    optionally also the bounding boxes in the original image.

    Returns:
        if return_boxes is False:
            [char28, char28, ...]
        if return_boxes is True:
            ([char28, ...], [(x,y,w,h), ...])
    """
    # Handle RGBA / RGB / grayscale
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Threshold: white background, dark ink -> invert so ink = 255
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))

    # sort left-to-right
    boxes.sort(key=lambda b: b[0])

    char_images = []
    for (x, y, w, h) in boxes:
        char_roi = thresh[y:y+h, x:x+w]  # binary

        side = max(w, h)
        square = np.zeros((side, side), dtype=np.uint8)
        x_offset = (side - w) // 2
        y_offset = (side - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = char_roi

        char28 = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        char28 = char28.astype("float32") / 255.0

        # If your training uses white-on-black vs black-on-white, adjust here:
        # char28 = 1.0 - char28

        char_images.append(char28)

    if return_boxes:
        return char_images, boxes
    return char_images


def segment_words_from_line(img, min_area=20, dilate=True,
                            gap_factor=1.5, return_boxes=False):
    """
    Segment a single handwritten line into words and characters.

    Returns:
        word_char_images: list of list of 28x28 images
            [
              [char28_word1_char1, char28_word1_char2, ...],
              [char28_word2_char1, ...],
              ...
            ]

        If return_boxes is True, also returns:
        word_char_boxes: list of list of (x,y,w,h) for each char in each word.
    """
    # Reuse the same preprocessing as segment_characters_from_word
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return ([], []) if return_boxes else []

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])

    # --- group boxes into words based on horizontal gap ---
    widths = [w for (_, _, w, _) in boxes]
    median_w = np.median(widths)
    word_gap_threshold = gap_factor * median_w  # gap > this => new word

    words_boxes = []
    current = [boxes[0]]

    for box in boxes[1:]:
        x, y, w, h = box
        prev_x, prev_y, prev_w, prev_h = current[-1]
        gap = x - (prev_x + prev_w)
        if gap > word_gap_threshold:
            # start new word
            words_boxes.append(current)
            current = [box]
        else:
            current.append(box)
    words_boxes.append(current)

    # --- build 28x28 images for each char in each word ---
    word_char_images = []
    word_char_boxes = []

    for word in words_boxes:
        char_imgs_this_word = []
        boxes_this_word = []
        for (x, y, w, h) in word:
            char_roi = thresh[y:y+h, x:x+w]

            side = max(w, h)
            square = np.zeros((side, side), dtype=np.uint8)
            x_offset = (side - w) // 2
            y_offset = (side - h) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = char_roi

            char28 = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
            char28 = char28.astype("float32") / 255.0
            # If you invert for training, do it here:
            # char28 = 1.0 - char28

            char_imgs_this_word.append(char28)
            boxes_this_word.append((x, y, w, h))

        word_char_images.append(char_imgs_this_word)
        word_char_boxes.append(boxes_this_word)

    if return_boxes:
        return word_char_images, word_char_boxes
    return word_char_images
