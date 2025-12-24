import numpy as np


# -----------------------------
# Forge a goofy binary classification realm
# -----------------------------
np.random.seed(42)

mystery_input = np.random.randn(200, 1)
truth_sauce_w = 2.0
truth_sauce_b = -0.5


# Secret recipe for labels: linear magic + squishy gate
vibes = truth_sauce_w * mystery_input + truth_sauce_b
prob_spells = 1 / (1 + np.exp(-vibes))
destiny_label = (prob_spells >= 0.5).astype(int)


# -----------------------------
# Random starting magic
# -----------------------------
chaos_weight = np.random.randn()
chaos_bias = 0.0

learning_potion = 0.1
ritual_rounds = 100


# -----------------------------
# Squishy Portal (Sigmoid)
# -----------------------------
def squishy_portal(signal):
    return 1 / (1 + np.exp(-signal))


# -----------------------------
# Heartbreak Score (Binary Cross Entropy)
# -----------------------------
def heartbreak_score(real, guess, sneeze=1e-8):
    guess = np.clip(guess, sneeze, 1 - sneeze)
    return -np.mean(real * np.log(guess) + (1 - real) * np.log(1 - guess))


# -----------------------------
# Training Ritual Begins
# -----------------------------
for round_num in range(ritual_rounds):
    # Channel the forward energy
    whisper = chaos_weight * mystery_input + chaos_bias
    prophecy = squishy_portal(whisper)

    # Feel the heartbreak
    ache = heartbreak_score(destiny_label, prophecy)

    # Calculate remorse (gradients)
    weight_tears = np.mean((prophecy - destiny_label) * mystery_input)
    bias_tears = np.mean(prophecy - destiny_label)

    # Apply emotional correction (update parameters)
    chaos_weight -= learning_potion * weight_tears
    chaos_bias -= learning_potion * bias_tears

    # Occasionally scream about progress
    if round_num % 10 == 0:
        print(f"Ritual {round_num:03d} | Heartbreak: {ache:.6f}")


# -----------------------------
# Ceremony is complete
# -----------------------------
print("\nSacred training complete.")
print(f"Final chaos_weight: {chaos_weight:.4f}")
print(f"Final chaos_bias: {chaos_bias:.4f}")
