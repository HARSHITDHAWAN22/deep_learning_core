import numpy as np


# -----------------------------
# Scaled Dot-Product Attention
# -----------------------------


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention(Q, K, V):
    """
    Computes scaled dot-product self-attention.
    """
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    output = np.dot(weights, V)
    return output, weights


# -----------------------------
# Example
# -----------------------------
np.random.seed(42)


# Sequence length = 3, embedding dim = 4
Q = np.random.randn(3, 4)
K = np.random.randn(3, 4)
V = np.random.randn(3, 4)

output, attn_weights = self_attention(Q, K, V)

print("Attention Output:\n", output)
print("\nAttention Weights:\n", attn_weights)
