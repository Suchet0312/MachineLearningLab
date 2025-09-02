import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 2a. Create CSV file
# -------------------------------
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Score":      [2, 4, 5, 4, 5, 7, 8, 9, 9, 10]
}
df = pd.DataFrame(data)
df.to_csv("ml_lab_data.csv", index=False)
print("Dataset created:\n", df)

# -------------------------------
# Step 2b. Regression Model (Pedhazur formula)
# -------------------------------
X = df["StudyHours"].values
Y = df["Score"].values
n = len(X)

x_mean = np.mean(X)
y_mean = np.mean(Y)

# slope (B1) and intercept (B0)
B1 = np.sum((X - x_mean) * (Y - y_mean)) / np.sum((X - x_mean)**2)
B0 = y_mean - B1 * x_mean

# predicted responses
Y_pred = B0 + B1 * X

# RMSE
rmse = np.sqrt(np.mean((Y - Y_pred)**2))

print("\nPedhazur Formula Method:")
print(f"Intercept (B0): {B0:.3f}")
print(f"Slope (B1): {B1:.3f}")
print(f"RMSE: {rmse:.3f}")
print("Predicted Responses:", Y_pred)

# -------------------------------
# Step 2c. Scatter Plot
# -------------------------------
plt.scatter(X, Y, color="red", label="Data Points")
plt.plot(X, Y_pred, color="blue", label="Regression Line")
plt.xlabel("Study Time (hours)")
plt.ylabel("Score")
plt.legend()
plt.show()

# -------------------------------
# Step 2d. Calculus / Normal Equation Method
# -------------------------------0
X_matrix = np.c_[np.ones(n), X]   # Add column of ones for intercept
Y_matrix = Y.reshape(-1, 1)

# Normal Equation: (X^T X)^(-1) X^T Y
theta = np.linalg.inv(X_matrix.T @ X_matrix) @ (X_matrix.T @ Y_matrix)

B0_calc, B1_calc = theta[0,0], theta[1,0]

print("\nCalculus (Normal Equation) Method:")
print(f"Intercept (B0): {B0_calc:.3f}")
print(f"Slope (B1): {B1_calc:.3f}")

# -------------------------------
# Step 2e. Compare coefficients
# -------------------------------
print("\nComparison of Methods:")
print(f"Pedhazur -> B0={B0:.3f}, B1={B1:.3f}")
print(f"Calculus -> B0={B0_calc:.3f}, B1={B1_calc:.3f}")

# -------------------------------
# Step 2f. Prediction for 10 hours
# -------------------------------
study_time = 10
predicted_score = B0 + B1 * study_time
print(f"\nPredicted score for {study_time} hours study: {predicted_score:.2f}")
