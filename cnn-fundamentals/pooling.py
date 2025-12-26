import numpy as np


# ===============================
# Max Pooling Operation
# ===============================
def max_pool(image, pool_size=2, stride=2):
    """Perform max pooling on a 2D image."""
    h_img, w_img = image.shape

    out_h = (h_img - pool_size) // stride + 1
    out_w = (w_img - pool_size) // stride + 1
    pooled = np.zeros((out_h, out_w), dtype=float)

    for r in range(out_h):
        for c in range(out_w):
            patch = image[
                r * stride:r * stride + pool_size,
                c * stride:c * stride + pool_size
            ]
            pooled[r, c] = np.max(patch)

    return pooled


# ===============================
# Average Pooling Operation
# ===============================
def avg_pool(image, pool_size=2, stride=2):
    """Perform average pooling on a 2D image."""
    h_img, w_img = image.shape

    out_h = (h_img - pool_size) // stride + 1
    out_w = (w_img - pool_size) // stride + 1
    pooled = np.zeros((out_h, out_w), dtype=float)

    for r in range(out_h):
        for c in range(out_w):
            patch = image[
                r * stride:r * stride + pool_size,
                c * stride:c * stride + pool_size
            ]
            pooled[r, c] = np.mean(patch)

    return pooled


# ===============================
# Example Run
# ===============================
if __name__ == "__main__":
    image = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [1, 2, 0, 1],
        [3, 1, 2, 4]
    ])

    print("Max Pooling:\n", max_pool(image))
    print("Average Pooling:\n", avg_pool(image))
