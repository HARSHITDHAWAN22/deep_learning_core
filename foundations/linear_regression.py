import numpy as np


# -----------------------------
# Generate a goofy synthetic dataset
# cosmic_truth = 3 * cosmic_input + 2 + some_noise
# -----------------------------
np.random.seed(42)

cosmic_input = np.random.rand(100, 1)
cosmic_truth = 3 * cosmic_input + 2 + 0.1 * np.random.randn(100, 1)


# -----------------------------
# Initialize the cosmic guesses
# -----------------------------
magic_weight = np.random.randn()
magic_bias = 0.0

coffee_strength = 0.1   # learning rate
lap_count = 100         # epochs


# -----------------------------
# Mean Squared Error (Pain Meter)
# -----------------------------
def pain_meter(actual, guess):
    return np.mean((actual - guess) ** 2)


# -----------------------------
# The Grand Training Ritual
# -----------------------------
for lap in range(lap_count):
    # Forward prophecy
    cosmic_prediction = magic_weight * cosmic_input + magic_bias

    # Measure the pain
    pain = pain_meter(cosmic_truth, cosmic_prediction)

    # Calculate how wrong we are (aka gradients)
    weight_sadness = -2 * np.mean(cosmic_input * (cosmic_truth - cosmic_prediction))
    bias_sadness = -2 * np.mean(cosmic_truth - cosmic_prediction)

    # Adjust the cosmic knobs (update params)
    magic_weight -= coffee_strength * weight_sadness
    magic_bias -= coffee_strength * bias_sadness

    # Broadcast progress
    if lap % 10 == 0:
        print(f"Lap {lap:03d} | Pain Level: {pain:.6f}")


# -----------------------------
# Final learned parameters
# -----------------------------
print("\nTraining ritual completed.")
print(f"Final magic weight: {magic_weight:.4f}")
print(f"Final magic bias: {magic_bias:.4f}")
