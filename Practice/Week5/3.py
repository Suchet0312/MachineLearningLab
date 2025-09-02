import pandas as pd
from sklearn.linear_model import LogisticRegression
# -------------------------------
# Step 1: Dataset
# -------------------------------
data2 = {"x1":[4,8,1,2,4,6],
         "x2":[1,-14,0,-1,7,-8],
         "y": [2,-14,1,-1,-7,-8]}
df2 = pd.DataFrame(data2)

# Convert y to binary (positive→1, negative→0)
df2["y_bin"] = (df2["y"] > 0).astype(int)
df2.to_csv("multi_logistic.csv", index=False)

# -------------------------------
# Step 2: Logistic Regression (Sklearn)
# -------------------------------
X2 = df2[["x1","x2"]].values
y2 = df2["y_bin"].values

model2 = LogisticRegression()
model2.fit(X2, y2)

print("Intercept:", model2.intercept_)
print("Coefficients:", model2.coef_)
print("Predictions:", model2.predict(X2))
print("Accuracy:", model2.score(X2, y2))
