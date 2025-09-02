import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Step 1: Dataset & Save to CSV
# -------------------------------
data = {"Hours": [1,2,3,4,5,6,7,8],
        "Pass":  [0,0,0,0,1,1,1,1]}
df = pd.DataFrame(data)
df.to_csv("study_pass.csv", index=False)

# -------------------------------
# Step 2: Sigmoid Function
# -------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -------------------------------
# Step 3: Logistic Regression with Gradient Descent
# -------------------------------
def logistic_regression_gd(X, y, lr=0.01, epochs=3):
    m = len(y)
    B0, B1 = 0.0, 0.0
    errors = []

    for epoch in range(epochs):
        for i in range(m):  # stochastic gradient descent
            xi, yi = X[i], y[i]
            z = B0 + B1 * xi
            pi = sigmoid(z)
            error = yi - pi

            # Update weights
            B0 += lr * error
            B1 += lr * error * xi

            loss = - (yi*np.log(pi+1e-9) + (1-yi)*np.log(1-pi+1e-9))
            errors.append(loss)

    return B0, B1, errors

# -------------------------------
# Step 4: Train Model
# -------------------------------
X = df["Hours"].values
y = df["Pass"].values

B0_gd, B1_gd, errors = logistic_regression_gd(X, y, lr=0.1, epochs=3)
y_pred_prob_gd = sigmoid(B0_gd + B1_gd*X)
y_pred_class_gd = (y_pred_prob_gd >= 0.5).astype(int)

acc_gd = accuracy_score(y, y_pred_class_gd)

# -------------------------------
# Step 5: Sklearn Logistic Regression
# -------------------------------
X_reshaped = X.reshape(-1,1)
model = LogisticRegression()
model.fit(X_reshaped, y)

y_pred_class_sk = model.predict(X_reshaped)
acc_sk = accuracy_score(y, y_pred_class_sk)

# -------------------------------
# Step 6: Predictions for 3.5 and 7.5 hrs
# -------------------------------
p_35 = sigmoid(B0_gd + B1_gd*3.5)
p_75 = sigmoid(B0_gd + B1_gd*7.5)

# -------------------------------
# Step 7: Results
# -------------------------------
print("=== Gradient Descent Model ===")
print(f"Intercept={B0_gd:.3f}, Coef={B1_gd:.3f}, Accuracy={acc_gd:.2f}")
print(f"Probability pass (3.5 hrs): {p_35:.3f}")
print(f"Probability pass (7.5 hrs): {p_75:.3f}")

print("\n=== Sklearn Model ===")
print(f"Intercept={model.intercept_[0]:.3f}, Coef={model.coef_[0][0]:.3f}, Accuracy={acc_sk:.2f}")

# -------------------------------
# Step 8: Plot Error vs Iteration
# -------------------------------
plt.plot(range(len(errors)), errors, marker="o", color="purple")
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Error vs Iteration (24 steps)")
plt.show()
