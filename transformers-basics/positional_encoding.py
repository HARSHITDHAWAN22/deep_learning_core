import numpy as np


# ===============================
# Positional Encoding (From Scratch)
# ===============================
def positional_encoding(seq_len, d_model):
    """Generate sinusoidal positional encodings for transformer models."""
    encoding = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            encoding[pos, i] = np.sin(angle)
            if i + 1 < d_model:
                encoding[pos, i + 1] = np.cos(angle)

    return encoding



# ===============================
# Example Run
# ===============================
if __name__ == "__main__":
    pe = positional_encoding(seq_len=5, d_model=6)
    print("Positional Encoding:\n", pe)
