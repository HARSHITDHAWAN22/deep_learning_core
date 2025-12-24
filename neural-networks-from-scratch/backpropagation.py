import numpy as np


np.random.seed(42)

X = np.random.randn(100, 2)
y = np.sum(X, axis=1, keepdims=True)

W1 = np.random.randn(2, 5)
b1 = np.zeros((1, 5))

W2 = np.random.randn(5, 1)
b2 = np.zeros((1, 1))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Forward pass
z1 = np.dot(X, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
y_pred = z2


# Loss
loss = mse(y, y_pred)
print("Loss:", loss)


# Backpropagation 
error_out = y_pred - y
grad_w2 = np.dot(a1.T, error_out) / len(X)
grad_b2 = np.mean(error_out, axis=0, keepdims=True)

error_hidden = np.dot(error_out, W2.T)
grad_z1 = error_hidden * relu_derivative(z1)
grad_w1 = np.dot(X.T, grad_z1) / len(X)
grad_b1 = np.mean(grad_z1, axis=0, keepdims=True)

print("Backpropagation completed")
