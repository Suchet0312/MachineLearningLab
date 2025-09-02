import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Year": [1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,
             1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,
             2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022],

    "Silver": [2715,2720,3105,3570,3955,4015,4794,6066,6755,6463,6646,8040,5489,7124,6335,7346,
               7346,8560,7615,7900,7215,7875,7695,11770,10675,17405,19520,23625,22165,27255,
               56900,56290,54030,43070,37825,36990,37825,41400,40600,63435,62572,55100],

    # Gold prices (from Q1, same years)
    "Gold": [1800,1645,1800,1970,2130,2140,2570,3130,3140,3200,3466,4334,4140,4598,4680,5160,
             4725,4045,4234,4400,4300,4990,5600,5850,7000,8400,10800,12500,14500,18500,
             26400,31050,29600,28006,26343,28623,29667,31438,35220,48651,50045,52950]
}

df = pd.DataFrame(data)
df.to_csv("gold_silver.csv", index=False)

# Load data
df = pd.read_csv("gold_silver.csv")   # <-- replace with your actual file name

# Extract features (X) and targets (y)
X = df[["Year"]].values        # Feature (independent variable) -> must be 2D
y_gold = df["Gold"].values     # Target 1 (dependent variable: Gold prices)
y_silver = df["Silver"].values # Target 2 (dependent variable: Silver prices)

# Train regression model for Gold
model_gold = LinearRegression()
model_gold.fit(X, y_gold)

# Train regression model for Silver
model_silver = LinearRegression()
model_silver.fit(X, y_silver)

# Predictions
y_pred_gold = model_gold.predict(X)
y_pred_silver = model_silver.predict(X)

# --- Plot Gold ---
plt.scatter(X, y_gold, color="gold", label="Actual Gold Prices")
plt.plot(X, y_pred_gold, color="red", label="Predicted Gold Prices")
plt.xlabel("Year")
plt.ylabel("Gold Price")
plt.title("Gold Price Prediction using Linear Regression")
plt.legend()
plt.show()

# --- Plot Silver ---
plt.scatter(X, y_silver, color="silver", label="Actual Silver Prices")
plt.plot(X, y_pred_silver, color="blue", label="Predicted Silver Prices")
plt.xlabel("Year")
plt.ylabel("Silver Price")
plt.title("Silver Price Prediction using Linear Regression")
plt.legend()
plt.show()

# Print model coefficients
print("Gold Price Model: y = {:.2f}*Year + {:.2f}".format(model_gold.coef_[0], model_gold.intercept_))
print("Silver Price Model: y = {:.2f}*Year + {:.2f}".format(model_silver.coef_[0], model_silver.intercept_))
