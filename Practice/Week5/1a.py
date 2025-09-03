import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------------
# Step 1: Dataset
# -------------------------------
data = {
    "Mother": [58, 62, 60, 64, 67, 70],
    "Daughter": [60, 60, 58, 60, 70, 72]
}
X = np.array(data["Mother"])
y = np.array(data["Daughter"])

# -------------------------------
# Step 2: Gradient Descent (simple)
# -------------------------------
def gradient_descent(X, y, lr=0.001, epochs=1000):
    B0, B1 = 0, 0
    n = len(X)
    errors = []

    for _ in range(epochs):
        y_pred = B0 + B1 * X
        error = y - y_pred

        # Update rule
        B0 -= lr * (-2 * np.sum(error)) / n
        B1 -= lr * (-2 * np.sum(X * error)) / n

        errors.append(np.mean(error**2))
    return B0, B1, errors

B0_gd, B1_gd, errors = gradient_descent(X, y)
y_pred_gd = B0_gd + B1_gd * X

# -------------------------------
# Step 3: Sklearn Linear Regression
# -------------------------------
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, y)

B0_sk, B1_sk = model.intercept_, model.coef_[0]
y_pred_sk = model.predict(X_reshaped)

# -------------------------------
# Step 4: Prediction for Mother=63
# -------------------------------
pred_gd = B0_gd + B1_gd * 63
pred_sk = model.predict([[63]])[0]

# -------------------------------
# Step 5: Results
# -------------------------------
print("=== Gradient Descent ===")
print(f"Intercept={B0_gd:.2f}, Slope={B1_gd:.2f}")
print(f"Prediction for mother=63: {pred_gd:.2f}")

print("\n=== Sklearn ===")
print(f"Intercept={B0_sk:.2f}, Slope={B1_sk:.2f}")
print(f"Prediction for mother=63: {pred_sk:.2f}")

# -------------------------------
# Step 6: Plot
# -------------------------------
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, y_pred_gd, color="red", label="GD Line")
plt.plot(X, y_pred_sk, color="green", linestyle="--", label="Sklearn Line")
plt.xlabel("Mother Height")
plt.ylabel("Daughter Height")
plt.legend()
plt.show()

# Error vs Iterations
plt.plot(errors, color="purple")
plt.xlabel("Iterations")
plt.ylabel("Error (MSE)")
plt.title("Error vs Iterations (GD)")
plt.show()
