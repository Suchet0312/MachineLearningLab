import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------------
# Step 1: Create CSV file (Year vs Price)
# -------------------------------
data = {
    "Year": [1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,
             1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,
             1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,
             2013,2014,2015,2016,2017,2018,2019,2020,2021,2022],
    "Price": [72,84,103,162,176,184,193,202,279,506,540,432,486,685,937,1330,
              1800,1645,1800,1970,2130,2140,2570,3130,3140,3200,3466,4334,4140,4598,4680,5160,
              4725,4045,4234,4400,4300,4990,5600,5850,7000,8400,10800,12500,14500,18500,26400,31050,
              29600,28006,26343,28623,29667,31438,35220,48651,50045,52950]
}

df = pd.DataFrame(data)
df.to_csv("gold_price.csv", index=False)

# -------------------------------
# Step 2: Prepare data
# -------------------------------
X = df[["Year"]].values  # feature
y = df["Price"].values   # target

# -------------------------------
# Step 3: Manual Regression (Normal Equation)
# -------------------------------
n = len(X)
X_matrix = np.c_[np.ones(n), X]  # Add intercept
y_matrix = y.reshape(-1, 1)

theta = np.linalg.inv(X_matrix.T @ X_matrix) @ (X_matrix.T @ y_matrix)
B0_manual, B1_manual = theta[0,0], theta[1,0]

# Predictions
y_pred_manual = B0_manual + B1_manual * X.flatten()

# Errors
mse_manual = mean_squared_error(y, y_pred_manual)
rmse_manual = np.sqrt(mse_manual)

# -------------------------------
# Step 4: Sklearn Regression
# -------------------------------
model = LinearRegression()
model.fit(X, y)
B0_sklearn = model.intercept_
B1_sklearn = model.coef_[0]

y_pred_sklearn = model.predict(X)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)
rmse_sklearn = np.sqrt(mse_sklearn)

# -------------------------------
# Step 5: Predictions for 2025
# -------------------------------
year = 2025
pred_manual = B0_manual + B1_manual * year
pred_sklearn = model.predict([[year]])[0]

# Convert per 10 grams → per gram
pred_manual_pergram = pred_manual / 10
pred_sklearn_pergram = pred_sklearn / 10

# -------------------------------
# Step 6: Results
# -------------------------------
print("Manual Method:")
print(f"Intercept B0 = {B0_manual:.3f}, Slope B1 = {B1_manual:.3f}")
print(f"MSE = {mse_manual:.2f}, RMSE = {rmse_manual:.2f}")

print("\nSklearn Method:")
print(f"Intercept B0 = {B0_sklearn:.3f}, Slope B1 = {B1_sklearn:.3f}")
print(f"MSE = {mse_sklearn:.2f}, RMSE = {rmse_sklearn:.2f}")

print(f"\nPrediction for 2025 (Manual): ₹{pred_manual:.2f} per 10g, ₹{pred_manual_pergram:.2f} per gram")
print(f"Prediction for 2025 (Sklearn): ₹{pred_sklearn:.2f} per 10g, ₹{pred_sklearn_pergram:.2f} per gram")

# -------------------------------
# Step 7: Plot
# -------------------------------
plt.scatter(X, y, color="red", label="Actual Data")
plt.plot(X, y_pred_manual, color="blue", label="Manual Regression")
plt.plot(X, y_pred_sklearn, color="green", linestyle="--", label="Sklearn Regression")
plt.xlabel("Year")
plt.ylabel("Gold Price (₹ per 10g)")
plt.legend()
plt.show()
