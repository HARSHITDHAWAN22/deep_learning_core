import numpy as np


np.random.seed(42)


# Input data
X = np.random.randn(100, 2)

#Later we will apply the algorithms that can intialize weight we will learn them later.
# Layer 1
W1 = np.random.randn(2, 5)
b1 = np.zeros((1, 5))


# Layer 2
W2 = np.random.randn(5, 1)
b2 = np.zeros((1, 1))


# ReLU
def relu(x):
    return np.maximum(0, x)


# Forward pass
def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    
    z2 = np.dot(a1, W2) + b2
    return z2

out = forward(X)
print("Output shape:", out.shape)
