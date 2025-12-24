import numpy as np


# ===============================
# Mean Squared Error (MSE)
# ===============================
def mse_loss(y_true, y_pred):
    """Calculate MSE: average of squared differences."""
    return np.mean((y_true - y_pred) ** 2)


# ===============================
# Binary Cross-Entropy Loss
# ===============================
def binary_cross_entropy(y_true, y_pred, epsilon=1e-8):
    """Calculate BCE loss for binary classification."""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ===============================
# Quick Test
# ===============================
if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([0.9, 0.2, 0.8, 0.7])

    print("MSE Loss:", mse_loss(y_true, y_pred))
    print("BCE Loss:", binary_cross_entropy(y_true, y_pred))
