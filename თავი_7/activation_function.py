import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. მონაცემების შექმნა
# -------------------------------------------------

torch.manual_seed(42)

# x მნიშვნელობები
x = torch.linspace(-2 * torch.pi, 2 * torch.pi, 300).reshape(-1, 1)

# სამიზნე ფუნქცია
y = torch.sin(x)


# -------------------------------------------------
# 2. მოდელი აქტივაციის ფუნქციის გარეშე
# -------------------------------------------------

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# -------------------------------------------------
# 3. მოდელი ReLU აქტივაციის ფუნქციით
# -------------------------------------------------

class NonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# -------------------------------------------------
# 4. მოდელების შექმნა
# -------------------------------------------------

linear_model = LinearModel()
nonlinear_model = NonLinearModel()


# -------------------------------------------------
# 5. Loss ფუნქცია
# -------------------------------------------------

criterion = nn.MSELoss()


# -------------------------------------------------
# 6. Optimizer-ები
# -------------------------------------------------

linear_optimizer = torch.optim.Adam(
    linear_model.parameters(),
    lr=0.01
)

nonlinear_optimizer = torch.optim.Adam(
    nonlinear_model.parameters(),
    lr=0.01
)


# -------------------------------------------------
# 7. მოდელების გაწვრთნა
# -------------------------------------------------

epochs = 3000

linear_losses = []
nonlinear_losses = []

for epoch in range(epochs):

    # ============================================
    # Linear Model
    # ============================================

    linear_optimizer.zero_grad()

    linear_prediction = linear_model(x)

    linear_loss = criterion(
        linear_prediction,
        y
    )

    linear_loss.backward()

    linear_optimizer.step()

    linear_losses.append(linear_loss.item())


    # ============================================
    # Non-Linear Model
    # ============================================

    nonlinear_optimizer.zero_grad()

    nonlinear_prediction = nonlinear_model(x)

    nonlinear_loss = criterion(
        nonlinear_prediction,
        y
    )

    nonlinear_loss.backward()

    nonlinear_optimizer.step()

    nonlinear_losses.append(
        nonlinear_loss.item()
    )


    # ყოველ 500 epoch-ზე შედეგის დაბეჭდვა
    if epoch % 500 == 0:
        print(
            f"Epoch: {epoch:4d} | "
            f"Linear Loss: {linear_loss.item():.6f} | "
            f"Non-linear Loss: {nonlinear_loss.item():.6f}"
        )


# -------------------------------------------------
# 8. საბოლოო პროგნოზების მიღება
# -------------------------------------------------

with torch.no_grad():

    linear_prediction = linear_model(x)

    nonlinear_prediction = nonlinear_model(x)


# -------------------------------------------------
# 9. შედეგების ვიზუალიზაცია
# -------------------------------------------------

plt.figure(figsize=(12, 7))

# რეალური sin(x)
plt.plot(
    x.numpy(),
    y.numpy(),
    linewidth=3,
    label="Real function: y = sin(x)"
)

# Linear მოდელი
plt.plot(
    x.numpy(),
    linear_prediction.numpy(),
    linewidth=3,
    linestyle="--",
    label="Linear network (no activation)"
)

# Non-linear მოდელი
plt.plot(
    x.numpy(),
    nonlinear_prediction.numpy(),
    linewidth=3,
    linestyle=":",
    color="red",
    label="Non-linear network (ReLU)"
)

plt.xlabel("x")
plt.ylabel("y")

plt.title(
    "Why Neural Networks Need Non-Linear Activation Functions"
)

plt.legend()

plt.grid(alpha=0.3)

plt.show()


# -------------------------------------------------
# 10. Loss-ის ვიზუალიზაცია
# -------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    linear_losses,
    label="Linear Network Loss"
)

plt.plot(
    nonlinear_losses,
    label="Non-linear Network Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title("Training Loss Comparison")

plt.legend()

plt.grid(alpha=0.3)

plt.yscale("log")

plt.show()