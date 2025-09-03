import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------
df = pd.read_csv("diabetes_csv.csv")

# ------------------------------------------------------
# 2. Remove unnecessary columns
# ------------------------------------------------------
if "id" in df.columns:  # IDs are identifiers, not features
    df.drop(columns=["id"], inplace=True)

# ------------------------------------------------------
# 3. Handle missing values
# ------------------------------------------------------
df = df.dropna()
print("After removing NaN:", df.shape)

# ------------------------------------------------------
# 4. Separate features (X) and target (y)
# ------------------------------------------------------
target_col = "Outcome"   # <-- explicitly set the target column
y = df[target_col].to_numpy()

X = df.drop(columns=[target_col])
X = pd.get_dummies(X, drop_first=True)  # encode categorical features
print("After encoding:", X.shape)

# ------------------------------------------------------
# 5. Convert to NumPy arrays
# ------------------------------------------------------
X_np = X.to_numpy()
print("Final NumPy shapes:", X_np.shape, y.shape)

# ------------------------------------------------------
# 6. Split into training and testing sets
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y, test_size=0.2, random_state=42
)

print("Training Data Shape:", X_train.shape, y_train.shape)
print("Testing Data Shape:", X_test.shape, y_test.shape)
