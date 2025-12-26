import numpy as np

# ===============================
# Simple RNN Cell (From Scratch)
# ===============================

np.random.seed(42)

input_size = 1
hidden_size = 4


# Initialize weights and bias
W_x = np.random.randn(input_size, hidden_size)
W_h = np.random.randn(hidden_size, hidden_size)
bias = np.zeros((1, hidden_size))


# ===============================
# Activation Function
# ===============================
def tanh(x):
    """Apply the hyperbolic tangent activation."""
    return np.tanh(x)


# ===============================
# RNN Cell Forward Step
# ===============================
def rnn_step(x_t, h_prev):
    """Compute the next hidden state for a single time step."""
    h_t = tanh(np.dot(x_t, W_x) + np.dot(h_prev, W_h) + bias)
    return h_t


# ===============================
# Example Sequence Run
# ===============================
if __name__ == "__main__":
    sequence = np.array([[1.0], [2.0], [3.0]])
    hidden_state = np.zeros((1, hidden_size))

    for t, x_t in enumerate(sequence):
        hidden_state = rnn_step(x_t.reshape(1, -1), hidden_state)
        print(f"Hidden state at time {t}:\n", hidden_state)
