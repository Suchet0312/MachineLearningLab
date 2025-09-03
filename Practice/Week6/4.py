import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Utilities
# ==========================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logloss_binary(y_true, p_pred, eps=1e-12):
    p = np.clip(p_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

# ==========================================
# Gradient Descent for Logistic Regression
# ==========================================
def logistic_regression_gd(x, y, lr=0.2, iters=60, w0_init=0.0, w1_init=0.0):
    w0, w1 = w0_init, w1_init
    w0_hist, w1_hist, loss_hist = [], [], []

    for t in range(iters):
        z = w0 + w1 * x
        p = sigmoid(z)

        # Gradients
        grad0 = np.mean(p - y)
        grad1 = np.mean((p - y) * x)

        # Update
        w0 -= lr * grad0
        w1 -= lr * grad1

        # Record
        w0_hist.append(w0)
        w1_hist.append(w1)
        loss_hist.append(logloss_binary(y, sigmoid(w0 + w1 * x)))

    return np.array(w0_hist), np.array(w1_hist), np.array(loss_hist)

# ==========================================
# Positive slope dataset
# ==========================================
x_pos = np.array([1, 2, 3, 4, 5])
y_pos = np.array([0, 0, 1, 1, 1])

w0_p, w1_p, ll_p = logistic_regression_gd(x_pos, y_pos)
print("Q4 Positive: final w0, w1, Log-loss =", w0_p[-1], w1_p[-1], ll_p[-1])

# Plot slope vs Log-loss (Positive)
plt.plot(w1_p, ll_p)
plt.xlabel("Slope (w1)")
plt.ylabel("Log-loss")
plt.title("Q4 (Positive slope) — Slope vs Log-loss")
plt.show()

# ==========================================
# Negative slope dataset
# ==========================================
x_neg = np.array([1, 2, 3, 4, 5])
y_neg = np.array([1, 1, 0, 0, 0])

w0_n, w1_n, ll_n = logistic_regression_gd(x_neg, y_neg)
print("Q4 Negative: final w0, w1, Log-loss =", w0_n[-1], w1_n[-1], ll_n[-1])

# Plot slope vs Log-loss (Negative)
plt.plot(w1_n, ll_n)
plt.xlabel("Slope (w1)")
plt.ylabel("Log-loss")
plt.title("Q4 (Negative slope) — Slope vs Log-loss")
plt.show()

# (Optional) Compare slope evolution
plt.plot(np.arange(len(w1_p)), w1_p, label="Positive")
plt.plot(np.arange(len(w1_n)), w1_n, label="Negative")
plt.xlabel("Iteration")
plt.ylabel("Slope (w1)")
plt.title("Q4 — Slope vs Iteration")
plt.legend()
plt.show()
