import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes_csv.csv")
print(df.shape)
print(df.head())

if 'id' in df.columns:
    df.drop(columns=['id'],inplace=True)

df = df.dropna()
print("After Dropping Missing Rows:", df.shape)

df = pd.get_dummies(df, drop_first=True)
print("After Creating Dummies:", df.shape)
df = df.fillna(df.mean(numeric_only=True))  

data_np = df.to_numpy()
print("NumPy Array Shape:", data_np.shape)


X = data_np[:, :-1]   
y = data_np[:, -1] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data Shape:", X_train.shape, y_train.shape)
print("Testing Data Shape:", X_test.shape, y_test.shape)