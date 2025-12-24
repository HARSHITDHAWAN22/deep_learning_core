import numpy as np

# ===============================
# Data Generation
# ===============================
np.random.seed(46)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)


# ===============================
# Parameter Initialization
# ===============================
w = np.random.randn()
b = 0.0
learning_rate = 0.1
epochs = 100


# ===============================
# Loss Function
# ===============================
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ===============================
# Training Loop
# ===============================
for epoch in range(epochs):
    y_pred = w * X + b
    loss = mse_loss(y, y_pred)

    dw = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.6f}")


# ===============================
# Results
# ===============================
print("\nTraining completed.")
print(f"Final weight (w): {w:.4f}")
print(f"Final bias (b): {b:.4f}")
