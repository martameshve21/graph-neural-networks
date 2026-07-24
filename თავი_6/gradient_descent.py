import numpy as np
import matplotlib.pyplot as plt

x_values = np.array([3, 7, 8])
y_actual = np.array([4, 5, 8])

w = 0.5

def loss(b):
    y_pred = w * x_values + b
    return np.sum((y_actual - y_pred) ** 2)

def dloss_db(b):
    y_pred = w * x_values + b
    residuals = y_actual - y_pred
    return np.sum(-2 * residuals)

b_range = np.linspace(-2, 5, 300)
loss_values = np.array([loss(b) for b in b_range])

b_points = [0, 1, 2, 2.5, 3.5, 4.5]

plt.figure(figsize=(10, 6))

plt.plot(b_range, loss_values, label="Loss function L(b)")

for b0 in b_points:
    current_loss = loss(b0)
    slope = dloss_db(b0)

    tangent_x = np.linspace(b0 - 0.4, b0 + 0.4, 20)
    tangent_y = current_loss + slope * (tangent_x - b0)

    plt.scatter(b0, current_loss)
    plt.plot(tangent_x, tangent_y, linestyle="--")

    plt.text(
        b0,
        current_loss + 2,
        f"b={b0}\nslope={slope:.1f}",
        fontsize=9
    )

plt.xlabel("b")
plt.ylabel("Loss")
plt.title("Loss function with tangent lines")
plt.grid(True)
plt.legend()
plt.show()
