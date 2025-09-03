import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
data = {
    "experience":[1.2,1.5,1.9,2.2,2.4,2.5,2.8,3.1,3.3,3.7,4.2,4.4],
    "salary":[1.7,2.4,2.3,3.1,3.7,4.2,4.4,6.1,5.4,5.7,6.4,6.2]
}
df = pd.DataFrame(data)
X = df["experience"].values
y = df["salary"].values

# ---------------------------------------------------------
# Q1 (a) MSE for fixed slopes and intercept b=1.1
# ---------------------------------------------------------
b = 1.1
slopes = [0.1, 1.5, 0.8]
print("\nQ1 (a): Fixed Intercept, Different Slopes")
for slope in slopes:
    y_pred = slope*X + b
    mse = mean_squared_error(y, y_pred)
    print(f"Slope={slope}, Intercept={b}, MSE={mse:.4f}")

# ---------------------------------------------------------
# Q1 (b) Sweep slope beta from 0 to 1.5
# ---------------------------------------------------------
slopes = np.arange(0,1.51,0.01)
mses = []
for slope in slopes:
    y_pred = slope*X + b
    mses.append(mean_squared_error(y,y_pred))

plt.plot(slopes,mses)
plt.xlabel("Slope (Beta)")
plt.ylabel("MSE")
plt.title("Q1 (b) Slope vs MSE")
plt.show()

# ---------------------------------------------------------
# Q1 (c) Sweep intercept b for fixed slope=1.0
# ---------------------------------------------------------
slope = 1.0
intercepts = np.arange(0,1.51,0.01)
mses_b = []
for inter in intercepts:
    y_pred = slope*X + inter
    mses_b.append(mean_squared_error(y,y_pred))

plt.plot(intercepts,mses_b)
plt.xlabel("Intercept (b)")
plt.ylabel("MSE")
plt.title("Q1 (c) Intercept vs MSE")
plt.show()

# ---------------------------------------------------------
# Q1 (d) Compare with Scikit Learn LinearRegression
# ---------------------------------------------------------
X_reshaped = X.reshape(-1,1)
model = LinearRegression()
model.fit(X_reshaped,y)
y_pred = model.predict(X_reshaped)
mse = mean_squared_error(y,y_pred)

print("\nQ1 (d): Scikit Learn Linear Regression")
print("Slope (coef):", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mse)

# ---------------------------------------------------------
# Q2: Stochastic Gradient Descent (manual implementation)
# ---------------------------------------------------------
alpha = 0.01
B0, B1 = 0, 0
errors = []

for epoch in range(5):   # 5 epochs
    for i in range(len(X)):   # loop through samples
        y_pred = B0 + B1*X[i]
        error = y[i] - y_pred
        B0 = B0 + alpha * error
        B1 = B1 + alpha * error * X[i]
        errors.append(error**2)

# Plot loss vs iteration
plt.plot(errors)
plt.xlabel("Iteration")
plt.ylabel("Squared Error")
plt.title("Q2 (a) Loss vs Iterations (SGD)")
plt.show()

print("\nQ2 (Manual SGD):")
print("Final B0:",B0," Final B1:",B1)

# ---------------------------------------------------------
# Q2 (b) Compare with Scikit Learn SGDRegressor
# ---------------------------------------------------------
sgd = SGDRegressor(max_iter=60, tol=None, eta0=0.01, learning_rate="constant")
sgd.fit(X.reshape(-1,1),y)

print("\nQ2 (Scikit Learn SGD):")
print("Sklearn SGD B0 (intercept):", sgd.intercept_[0])
print("Sklearn SGD B1 (coef):", sgd.coef_[0])
