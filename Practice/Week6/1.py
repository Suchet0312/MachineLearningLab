import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "experience":[1.2,1.5,1.9,2.2,2.4,2.5,2.8,3.1,3.3,3.7,4.2,4.4],
    "salary":[1.7,2.4,2.3,3.1,3.7,4.2,4.4,6.1,5.4,5.7,6.4,6.2]
}
df = pd.DataFrame(data)
X = df["experience"].values
y = df["salary"].values

# (a) MSE for specific slopes
b = 1.1
slopes = [0.1, 1.5, 0.8]
for slope in slopes:
    y_pred = slope*X + b
    mse = mean_squared_error(y, y_pred)
    print(f"Slope={slope}, Intercept={b}, MSE={mse:.4f}")

# (b) Beta sweep
slopes = np.arange(0,1.51,0.01)
mses = []
for slope in slopes:
    y_pred = slope*X + b
    mses.append(mean_squared_error(y,y_pred))
plt.plot(slopes,mses)
plt.xlabel("Slope (Beta)")
plt.ylabel("MSE")
plt.title("Slope vs MSE")
plt.show()

# (c) Intercept sweep
slope = 1.0
intercepts = np.arange(0,1.51,0.01)
mses_b = []
for inter in intercepts:
    y_pred = slope*X + inter
    mses_b.append(mean_squared_error(y,y_pred))
plt.plot(intercepts,mses_b)
plt.xlabel("Intercept (b)")
plt.ylabel("MSE")
plt.title("Intercept vs MSE")
plt.show()

# (d) Scikit Learn
X_reshaped = X.reshape(-1,1)
model = LinearRegression()
model.fit(X_reshaped,y)
y_pred = model.predict(X_reshaped)
mse = mean_squared_error(y,y_pred)
print("Scikit Learn Results:")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mse)
