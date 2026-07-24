
import matplotlib.pyplot as plt
b_values = [    0, 0.25, 0.5, 0.75, 1.0,  1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

w = 0.5
x = [3, 7, 8]
y_actual = [4, 5, 8]
sum_squared_errors = []

for b in b_values:
    squared_errors = []
    for xi, yi in zip(x, y_actual):
        y_pred = (w * xi + b) 
        error = (y_pred - yi) ** 2
        squared_errors.append(error)
    total_error = sum(squared_errors)
    sum_squared_errors.append(total_error)

    print(f"b = {b}, total_error = {total_error}")

# Plot
plt.plot(b_values, sum_squared_errors, marker='o')
plt.xlabel("b")
plt.ylabel("Sum of Squared Errors")
plt.title("Least Squares Error vs b")

plt.grid(True)

plt.show()


