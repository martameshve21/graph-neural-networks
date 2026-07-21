import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. ნამდვილი ფუნქცია
# -----------------------------
def true_function(x):
    return torch.sin(x) + 0.3 * torch.sin(3 * x)

# -----------------------------
# 2. მონაცემების შექმნა
# -----------------------------
x = torch.linspace(-2 * np.pi, 2 * np.pi, 300).reshape(-1, 1)
y = true_function(x)

# -----------------------------
# 3. პატარა ნეირონული ქსელი
# -----------------------------
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

# -----------------------------
# 4. Loss და optimizer
# -----------------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 5. Training
# -----------------------------
epochs = 3000
losses = []

for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

# -----------------------------
# 6. შედეგის დახატვა
# -----------------------------
with torch.no_grad():
    y_pred = model(x)

plt.figure(figsize=(10, 5))
plt.plot(x.numpy(), y.numpy(), label="True function")
plt.plot(x.numpy(), y_pred.numpy(), label="Neural network approximation")
plt.legend()
plt.title("Universal Approximation Theorem Visualization")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# -----------------------------
# 7. Loss-ის დახატვა
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()