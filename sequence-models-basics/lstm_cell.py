import numpy as np


# ===============================
# LSTM Cell (Forward Pass)
# ===============================

np.random.seed(42)

input_size = 1
hidden_size = 4


# LSTM gate weights (forget, input, output, cell)
W_f = np.random.randn(input_size + hidden_size, hidden_size)  # Forget gate
W_i = np.random.randn(input_size + hidden_size, hidden_size)  # Input gate  
W_o = np.random.randn(input_size + hidden_size, hidden_size)  # Output gate
W_c = np.random.randn(input_size + hidden_size, hidden_size)  # Cell gate


# Biases
b_f = np.zeros((1, hidden_size))
b_i = np.zeros((1, hidden_size))
b_o = np.zeros((1, hidden_size))
b_c = np.zeros((1, hidden_size))


# ===============================
# Helper Functions
# ===============================
def sigmoid(x):
    """Sigmoid activation for gates."""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh activation for cell state."""
    return np.tanh(x)


# ===============================
# LSTM Cell Forward Pass
# ===============================
def lstm_step(x_t, h_prev, c_prev):
    """Single LSTM cell forward pass."""
    # Concatenate input and previous hidden state
    concat = np.concatenate((h_prev, x_t), axis=1)

  
    # Gate computations
    f_t = sigmoid(np.dot(concat, W_f) + b_f)  # Forget gate
    i_t = sigmoid(np.dot(concat, W_i) + b_i)  # Input gate
    o_t = sigmoid(np.dot(concat, W_o) + b_o)  # Output gate
    c_hat = tanh(np.dot(concat, W_c) + b_c)  # Cell candidate

  
    # Cell and hidden state updates
    c_t = f_t * c_prev + i_t * c_hat
    h_t = o_t * tanh(c_t)

    return h_t, c_t


# ===============================
# Example Run
# ===============================
if __name__ == "__main__":
    x_t = np.array([[1.0]])
    h = np.zeros((1, hidden_size))
    c = np.zeros((1, hidden_size))

    h_new, c_new = lstm_step(x_t, h, c)
    print("Hidden state:\n", h_new)
    print("Cell state:\n", c_new)
