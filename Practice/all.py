import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

# -----------------------------------
# 1. LINEAR REGRESSION (Exam Score vs Study Hours)
# -----------------------------------
data1 = {"Hours":[1,2,3,4,5,6,7,8,9,10], "Score":[2,4,5,4,5,7,8,9,9,10]}
df1 = pd.DataFrame(data1)
X1, y1 = df1["Hours"].values, df1["Score"].values
n = len(X1)

# --- Pedhazur formula
x_mean, y_mean = np.mean(X1), np.mean(y1)
B1 = np.sum((X1-x_mean)*(y1-y_mean))/np.sum((X1-x_mean)**2)
B0 = y_mean - B1*x_mean
Y_pred1 = B0 + B1*X1
rmse1 = np.sqrt(np.mean((y1-Y_pred1)**2))

print("=== Linear Regression (Study vs Score) ===")
print(f"Pedhazur: B0={B0:.2f}, B1={B1:.2f}, RMSE={rmse1:.2f}")

# --- Normal Equation
X1_mat = np.c_[np.ones(n), X1]
theta = np.linalg.inv(X1_mat.T @ X1_mat) @ (X1_mat.T @ y1.reshape(-1,1))
print(f"Normal Eq: B0={theta[0,0]:.2f}, B1={theta[1,0]:.2f}")

# --- Sklearn
model1 = LinearRegression().fit(X1.reshape(-1,1), y1)
print(f"Sklearn: B0={model1.intercept_:.2f}, B1={model1.coef_[0]:.2f}")

# -----------------------------------
# 2. LINEAR REGRESSION (Gold Price Prediction)
# -----------------------------------
years = np.arange(1965, 2023)
prices = np.linspace(72, 52950, len(years))  # simplified sample
df2 = pd.DataFrame({"Year":years, "Price":prices})
X2, y2 = df2[["Year"]].values, df2["Price"].values

# --- Normal Equation
n2 = len(X2)
X2_mat = np.c_[np.ones(n2), X2]
theta2 = np.linalg.inv(X2_mat.T @ X2_mat) @ (X2_mat.T @ y2.reshape(-1,1))
B0_g, B1_g = theta2[0,0], theta2[1,0]

# --- Sklearn
model2 = LinearRegression().fit(X2, y2)
print("\n=== Gold Price Regression ===")
print(f"Manual: B0={B0_g:.2f}, B1={B1_g:.2f}")
print(f"Sklearn: B0={model2.intercept_:.2f}, B1={model2.coef_[0]:.2f}")
print("Prediction 2025:", model2.predict([[2025]])[0])

# -----------------------------------
# 3. LINEAR REGRESSION (Mother vs Daughter Heights, Gradient Descent)
# -----------------------------------
X3 = np.array([58,62,60,64,67,70])
y3 = np.array([60,60,58,60,70,72])

def gradient_descent(X, y, lr=0.001, epochs=1000):
    B0, B1 = 0, 0
    n = len(X)
    for _ in range(epochs):
        y_pred = B0 + B1*X
        error = y - y_pred
        B0 += lr * np.sum(error)/n
        B1 += lr * np.sum(X*error)/n
    return B0, B1

B0_gd, B1_gd = gradient_descent(X3, y3)
print("\n=== Gradient Descent (Mother vs Daughter) ===")
print(f"GD: B0={B0_gd:.2f}, B1={B1_gd:.2f}")
model3 = LinearRegression().fit(X3.reshape(-1,1), y3)
print(f"Sklearn: B0={model3.intercept_:.2f}, B1={model3.coef_[0]:.2f}")

# -----------------------------------
# 4. LOGISTIC REGRESSION (Pass/Fail vs Hours)
# -----------------------------------
data4 = {"Hours":[1,2,3,4,5,6,7,8], "Pass":[0,0,0,0,1,1,1,1]}
df4 = pd.DataFrame(data4)
X4, y4 = df4["Hours"].values, df4["Pass"].values

def sigmoid(z): return 1/(1+np.exp(-z))

# Simple gradient descent for logistic
B0, B1 = 0.0, 0.0
for epoch in range(100):
    for xi, yi in zip(X4, y4):
        pi = sigmoid(B0+B1*xi)
        error = yi - pi
        B0 += 0.01*error
        B1 += 0.01*error*xi

y_pred_prob = sigmoid(B0 + B1*X4)
y_pred_class = (y_pred_prob>=0.5).astype(int)
print("\n=== Logistic Regression (Study vs Pass) ===")
print(f"GD: Intercept={B0:.2f}, Coef={B1:.2f}, Accuracy={accuracy_score(y4,y_pred_class):.2f}")

# Sklearn
model4 = LogisticRegression().fit(X4.reshape(-1,1), y4)
print(f"Sklearn: Intercept={model4.intercept_[0]:.2f}, Coef={model4.coef_[0][0]:.2f}, Accuracy={model4.score(X4.reshape(-1,1), y4):.2f}")

# -----------------------------------
# 5. MULTIVARIATE LOGISTIC (2 features)
# -----------------------------------
data5 = {"x1":[4,8,1,2,4,6],"x2":[1,-14,0,-1,7,-8],"y":[2,-14,1,-1,-7,-8]}
df5 = pd.DataFrame(data5)
df5["y_bin"] = (df5["y"]>0).astype(int)

X5, y5 = df5[["x1","x2"]].values, df5["y_bin"].values
model5 = LogisticRegression().fit(X5, y5)
print("\n=== Multivariate Logistic Regression ===")
print("Intercept:", model5.intercept_)
print("Coefficients:", model5.coef_)
print("Accuracy:", model5.score(X5, y5))
