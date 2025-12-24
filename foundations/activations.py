import numpy as np


# -----------------------------
# Squishy Gate (Sigmoid)
# -----------------------------
def squishy_gate(zap):
    """
    Squishy gate (Sigmoid activation function).
    """
    return 1 / (1 + np.exp(-zap))


def squishy_boost(zap):
    """
    Derivative of the squishy gate.
    """
    squish = squishy_gate(zap)
    return squish * (1 - squish)


# -----------------------------
# ZapCut (ReLU)
# -----------------------------
def zapcut(zap):
    """
    ZapCut (ReLU activation function).
    """
    return np.maximum(0, zap)


def zapcut_boost(zap):
    """
    Derivative of ZapCut.
    """
    return np.where(zap > 0, 1, 0)


# -----------------------------
# MoodCurve (Tanh)
# -----------------------------
def moodcurve(zap):
    """
    Mood curve (Tanh activation function).
    """
    return np.tanh(zap)


def moodcurve_boost(zap):
    """
    Derivative of the mood curve.
    """
    return 1 - np.tanh(zap) ** 2


# -----------------------------
# Quick check (optional)
# -----------------------------
if __name__ == "__main__":
    sample_wave = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    print("Raw vibe:", sample_wave)
    print("Squishy Gate:", squishy_gate(sample_wave))
    print("ZapCut:", zapcut(sample_wave))
    print("MoodCurve:", moodcurve(sample_wave))
