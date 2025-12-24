import numpy as np

np.random.seed(48)


# Data
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1]).reshape(-1, 1)


# Layers
w1 = np.random.randn(2, 8)
b1 = np.zeros((1, 8))

w2 = np.random.randn(8, 1)
b2 = np.zeros((1, 1))

lr = 0.01
epochs = 200

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X,w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1,w2) + b2
    y_pred = z2

    loss = mse_loss(y, y_pred)

  
    # Backprop
    error2 = y_pred - y
    dw2 = np.dot(a1.T, error2)/len(X)
    db2 = np.mean(error2,axis=0, keepdims=True)

    error1= np.dot(error2, w2.T)
    dz1= error1 * relu_grad(z1)
    dw1= np.dot(X.T,dz1)/len(X)
    db1= np.mean(dz1,axis=0, keepdims=True)

  
    # Update weights
    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

    if epoch % 20 == 0:
        print(f"Epoch{epoch} | Loss:{loss:.4f}")
