import numpy as np

np.random.seed(0)


# Tiny dataset
X = np.random.randn(10, 2)
y = (X[:, 0] ** 2 + X[:, 1] ** 2).reshape(-1, 1)

w1 = np.random.randn(2, 20)
b1 = np.zeros((1, 20))

w2 = np.random.randn(20, 1)
b2 = np.zeros((1, 1))

lr = 0.01
epochs = 500


def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

for epoch in range(epochs):
    z1 = np.dot(X, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    pred = z2

    loss = mse(y, pred)

    dz2 = pred - y
    dw2 = np.dot(a1.T, dz2) / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * relu_grad(z1)
    dw1 = np.dot(X.T, dz1) / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")
