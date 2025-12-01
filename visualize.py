import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def show_image(img, title="Raw Image"):
    """Display the original grayscale image."""
    plt.figure(figsize=(3, 3))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_edges(edges, title="Edge Map"):
    """Display the binary edge map."""
    plt.figure(figsize=(3, 3))
    plt.imshow(edges, cmap="gray_r")
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_segments(img, edges, grid_rows=4, grid_cols=4, title="Segmentation Grid"):
    """
    Shows:
    - the raw image
    - the edge map
    - the grid segmentation overlay (heatmap = edge density)
    """
    H, W = edges.shape
    seg_h = H // grid_rows
    seg_w = W // grid_cols

    # Compute density heatmap
    heatmap = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    for r in range(grid_rows):
        for c in range(grid_cols):
            r_start = r * seg_h
            c_start = c * seg_w
            r_end = H if r == grid_rows - 1 else (r_start + seg_h)
            c_end = W if c == grid_cols - 1 else (c_start + seg_w)

            segment = edges[r_start:r_end, c_start:c_end]
            heatmap[r, c] = segment.mean()

    # ───────────────────────────────────────────────
    # Figure layout: Raw image | Edges | Segmented
    # ───────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2])

    # Raw image
    ax0 = plt.subplot(gs[0])
    ax0.imshow(img, cmap="gray")
    ax0.set_title("Original")
    ax0.axis("off")

    # Edge map
    ax1 = plt.subplot(gs[1])
    ax1.imshow(edges, cmap="gray_r")
    ax1.set_title("Edge Map")
    ax1.axis("off")

    # Segmented heatmap
    ax2 = plt.subplot(gs[2])
    ax2.imshow(heatmap, cmap="plasma", vmin=0, vmax=1)
    ax2.set_title(title)
    ax2.set_xticks(np.arange(grid_cols))
    ax2.set_yticks(np.arange(grid_rows))

    # Draw the white gridlines
    ax2.set_xticks(np.arange(-0.5, grid_cols, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, grid_rows, 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=1.5)
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.show()


def show_feature_vector(features, grid_rows=4, grid_cols=4, title="Feature Vector Heatmap"):
    """
    Displays the feature vector (flattened) as a heatmap.
    Perfect for debugging ANN inputs.
    """
    feat_map = features.reshape(grid_rows, grid_cols)

    plt.figure(figsize=(4, 4))
    plt.imshow(feat_map, cmap="plasma", vmin=0, vmax=1)
    plt.title(title)

    # Gridlines
    plt.xticks(np.arange(grid_cols))
    plt.yticks(np.arange(grid_rows))
    plt.xticks(np.arange(-0.5, grid_cols, 1), minor=True)
    plt.yticks(np.arange(-0.5, grid_rows, 1), minor=True)
    plt.grid(which="minor", color="white", linewidth=1.5)

    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.show()
