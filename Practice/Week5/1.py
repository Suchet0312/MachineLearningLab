import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------------
# Step 1: Dataset & Save to CSV
# -------------------------------
data = {
    "Mother": [58, 62, 60, 64, 67, 70],
    "Daughter": [60, 60, 58, 60, 70, 72]
}
df = pd.DataFrame(data)
df.to_csv("mother_daughter.csv", index=False)

# -------------------------------
# Step 2: Gradient Descent Function
# -------------------------------
def gradient_descent(X, y, lr=0.001, epochs=4, batch_size=6):
    n = len(X)
    B0, B1 = 0.0, 0.0  # Initialize coefficients
    errors = []

    # Total iterations = epochs * (n/batch_size)
    for epoch in range(epochs):
        for i in range(0, n, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            y_pred = B0 + B1 * X_batch
            error = y_batch - y_pred

            # Gradient updates
            B0 -= lr * (-2 * np.sum(error)) / len(X_batch)
            B1 -= lr * (-2 * np.sum(X_batch * error)) / len(X_batch)

            mse = np.mean(error**2)
            errors.append(mse)

    return B0, B1, errors

# -------------------------------
# Step 3: Prepare Data
# -------------------------------
X = df["Mother"].values
y = df["Daughter"].values

# Run Gradient Descent
B0_gd, B1_gd, errors = gradient_descent(X, y, lr=0.001, epochs=4, batch_size=6)
y_pred_gd = B0_gd + B1_gd * X

# Metrics
mse_gd = mean_squared_error(y, y_pred_gd)
rmse_gd = np.sqrt(mse_gd)
log_loss_gd = np.mean(np.log(1 + (y - y_pred_gd) ** 2))  # custom "log loss" for regression

# -------------------------------
# Step 4: Sklearn Regression
# -------------------------------
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, y)

B0_sklearn, B1_sklearn = model.intercept_, model.coef_[0]
y_pred_sklearn = model.predict(X_reshaped)

mse_sk = mean_squared_error(y, y_pred_sklearn)
rmse_sk = np.sqrt(mse_sk)

# -------------------------------
# Step 5: Predict Daughter’s height for mother=63
# -------------------------------
pred_gd = B0_gd + B1_gd * 63
pred_sk = model.predict([[63]])[0]

# -------------------------------
# Step 6: Results
# -------------------------------
print("=== Gradient Descent Model ===")
print(f"Intercept={B0_gd:.2f}, Slope={B1_gd:.2f}")
print(f"MSE={mse_gd:.2f}, RMSE={rmse_gd:.2f}, LogLoss={log_loss_gd:.2f}")
print(f"Prediction for mother=63: {pred_gd:.2f}")

print("\n=== Sklearn Model ===")
print(f"Intercept={B0_sklearn:.2f}, Slope={B1_sklearn:.2f}")
print(f"MSE={mse_sk:.2f}, RMSE={rmse_sk:.2f}")
print(f"Prediction for mother=63: {pred_sk:.2f}")

# -------------------------------
# Step 7: Plot Graphs
# -------------------------------
plt.figure(figsize=(12,5))

# Scatter + Best Fit Line
plt.subplot(1,2,1)
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred_gd, color="red", label="GD Best Fit")
plt.plot(X, y_pred_sklearn, color="green", linestyle="--", label="Sklearn Best Fit")
plt.xlabel("Mother Height")
plt.ylabel("Daughter Height")
plt.legend()

# Error vs Iterations
plt.subplot(1,2,2)
plt.plot(range(len(errors)), errors, marker="o", color="purple")
plt.xlabel("Iterations")
plt.ylabel("Error (MSE)")
plt.title("Error vs Iterations")

plt.tight_layout()
plt.show() 