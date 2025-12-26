import numpy as np

# ===============================
# Toy Sequence Prediction (RNN)
# ===============================

np.random.seed(42)


# Simple sequence: y = x + 1
inputs = np.array([[1], [2], [3], [4]])
targets = np.array([[2], [3], [4], [5]])

input_dim = 1
hidden_dim = 5
output_dim = 1


# Model parameters
W_x = np.random.randn(input_dim, hidden_dim)     
W_h = np.random.randn(hidden_dim, hidden_dim)     
W_y = np.random.randn(hidden_dim, output_dim)     

b_h = np.zeros((1, hidden_dim))                   
b_y = np.zeros((1, output_dim))                   


# ===============================
# Activation & Forward Step
# ===============================
def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)


def rnn_forward(x_t, h_prev):
    """RNN forward pass with output prediction."""
    # Hidden state update
    h_next = tanh(np.dot(x_t, W_x) + np.dot(h_prev, W_h) + b_h)
    
    # Output prediction
    y_pred = np.dot(h_next, W_y) + b_y
    
    return h_next, y_pred


# ===============================
# Run Sequence Prediction
# ===============================
if __name__ == "__main__":
    hidden_state = np.zeros((1, hidden_dim))
    
    print("RNN Sequence Predictions:")
    print("-" * 30)
    
    for t in range(len(inputs)):
        x_t = inputs[t].reshape(1, -1)
        hidden_state, prediction = rnn_forward(x_t, hidden_state)
        
        print(f"Input: {inputs[t][0]:.0f} → Predicted: {prediction[0][0]:.2f}")
        print(f"Target:  {targets[t][0]:.0f}")
        print()
