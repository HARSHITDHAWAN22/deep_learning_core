import torch
import torch.nn as nn


# -----------------------------
# Mini Transformer Block
# -----------------------------


class MiniTransformerBlock(nn.Module):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# -----------------------------
# Example
# -----------------------------
seq_len = 5
batch_size = 2
d_model = 32

x = torch.randn(seq_len, batch_size, d_model)
block = MiniTransformerBlock(d_model=d_model)

out = block(x)
print("Output shape:", out.shape)
