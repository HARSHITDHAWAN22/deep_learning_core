import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Tiny CNN model
# -----------------------------
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # conv + ReLU + max-pool
        x = x.view(x.size(0), -1)                  # flatten batch
        x = self.fc(x)                             # final class scores
        return x


# -----------------------------
# One dummy training step
# -----------------------------
model = TinyCNN()
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)


# Fake data: batch of 16 grayscale 28x28 images and labels 0–9
x_batch = torch.randn(16, 1, 28, 28)
y_batch = torch.randint(0, 10, (16,))


# Forward pass
logits = model(x_batch)
loss = loss_fn(logits, y_batch)


# Backward + update
opt.zero_grad()
loss.backward()
opt.step()

print("Training step done. Loss:", loss.item())
