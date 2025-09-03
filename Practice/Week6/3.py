import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Utility: MSE
# ==========================================
def mse_linear(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ==========================================
# Gradient Descent for Linear Regression
# ==========================================
def linear_regression_gd(x, y, lr=0.02, iters=60, b0_init=0.0, b1_init=0.0):
    n = len(x)
    b0, b1 = b0_init, b1_init
    b0_hist, b1_hist, loss_hist = [], [], []

    for t in range(iters):
        y_hat = b0 + b1 * x
        err = y_hat - y

        # Gradients
        db0 = (2.0 / n) * np.sum(err)
        db1 = (2.0 / n) * np.sum(err * x)

        # Update
        b0 -= lr * db0
        b1 -= lr * db1

        # Record
        b0_hist.append(b0)
        b1_hist.append(b1)
        loss_hist.append(mse_linear(y, b0 + b1 * x))

    return np.array(b0_hist), np.array(b1_hist), np.array(loss_hist)

# ==========================================
# Positive slope dataset
# ==========================================
x_pos = np.array([1, 2, 4, 3, 5])
y_pos = np.array([1, 3, 3, 2, 5])

b0_p, b1_p, loss_p = linear_regression_gd(x_pos, y_pos)
print("Q3 Positive: final b0, b1, MSE =", b0_p[-1], b1_p[-1], loss_p[-1])

# Plot slope vs MSE (Positive)
plt.plot(b1_p, loss_p)
plt.xlabel("Slope (b1)")
plt.ylabel("MSE")
plt.title("Q3 (Positive slope) — Slope vs MSE")
plt.show()

# ==========================================
# Negative slope dataset
# ==========================================
x_neg = np.array([1, 2, 3, 4, 5])
y_neg = np.array([10, 8, 6, 4, 2])

b0_n, b1_n, loss_n = linear_regression_gd(x_neg, y_neg)
print("Q3 Negative: final b0, b1, MSE =", b0_n[-1], b1_n[-1], loss_n[-1])

# Plot slope vs MSE (Negative)
plt.plot(b1_n, loss_n)
plt.xlabel("Slope (b1)")
plt.ylabel("MSE")
plt.title("Q3 (Negative slope) — Slope vs MSE")
plt.show()

# (Optional) Compare slope evolution
plt.plot(np.arange(len(b1_p)), b1_p, label="Positive")
plt.plot(np.arange(len(b1_n)), b1_n, label="Negative")
plt.xlabel("Iteration")
plt.ylabel("Slope (b1)")
plt.title("Q3 — Slope vs Iteration")
plt.legend()
plt.show()
