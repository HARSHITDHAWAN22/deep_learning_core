import numpy as np


# ===============================
# Basic 2D Convolution
# ===============================
def conv2d(image, kernel, stride=1, padding=0):
    """Simple 2D convolution with stride and padding."""
    h_img, w_img = image.shape
    h_k, w_k = kernel.shape

    # Apply zero padding if needed
    if padding > 0:
        image_padded = np.pad(
            image,
            pad_width=((padding, padding), (padding, padding)),
            mode="constant"
        )
    else:
        image_padded = image

  
    # Output feature map size
    out_h = (h_img - h_k + 2 * padding) // stride + 1
    out_w = (w_img - w_k + 2 * padding) // stride + 1
    feature_map = np.zeros((out_h, out_w), dtype=float)

  
    # Convolution: slide kernel over image
    for r in range(out_h):
        for c in range(out_w):
            r_start = r * stride
            c_start = c * stride
            patch = image_padded[
                r_start:r_start + h_k,
                c_start:c_start + w_k
            ]
            feature_map[r, c] = np.sum(patch * kernel)

    return feature_map


# ===============================
# Demo Example
# ===============================
if __name__ == "__main__":
    image = np.array([
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 1, 0, 2],
        [2, 3, 1, 0]
    ])

    kernel = np.array([
        [1, 0],
        [0, -1]
    ])

    result = conv2d(image, kernel, stride=1, padding=1)
    print("Convolution result:\n", result)
