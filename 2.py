import csv
import numpy as np
import matplotlib.pyplot as plt
# -------------------------
# Step 1: Create CSV file with data
# -------------------------

data = [
[1, 2],
[2, 4],
[3, 5],
[4, 4],
[5, 6],
[6, 7],
[7, 8],
[8, 9],
[9, 10],
[10, 9]
]
with open('study_scores.csv', 'w', newline='') as file:
    writer = csv.   writer(file)
    writer.writerow(['StudyHours', 'Score'])
    writer.writerows(data)
print("CSV file 'study_scores.csv' created.\n")
# -------------------------
# Step 2: Load data from CSV
# -------------------------
x = []
y = []
with open('study_scores.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        x.append(float(row['StudyHours']))
        y.append(float(row['Score']))
        x = np.array(x)
        y = np.array(y)
# -------------------------
# Step 3: Linear Regression using Pedhazur's formula (intuitive)
# Calculate slope (B1) and intercept (B0)
# -------------------------
x_mean = np.mean(x)
y_mean = np.mean(y)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean)**2)
B1_pedhazur = numerator / denominator
B0_pedhazur = y_mean - B1_pedhazur * x_mean
print(f"Pedhazur's formula coefficients:")
print(f" B0 (intercept) = {B0_pedhazur:.4f}")

print(f" B1 (slope) = {B1_pedhazur:.4f}\n")
# -------------------------
# Step 4: Linear Regression using calculus (partial derivatives)
# Solve linear system for B0 and B1
# -------------------------
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x**2)
A = np.array([[n, sum_x],
[sum_x, sum_x2]])
b = np.array([sum_y, sum_xy])
B0_calc, B1_calc = np.linalg.solve(A, b)
print(f"Calculus method coefficients:")
print(f" B0 (intercept) = {B0_calc:.4f}")
print(f" B1 (slope) = {B1_calc:.4f}\n")
# -------------------------
# Step 5: Predict responses using Pedhazur's coefficients
# Calculate RMSE between actual y and predicted y
# -------------------------
y_pred = B0_pedhazur + B1_pedhazur * x
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
print(f"RMSE (Root Mean Square Error) = {rmse:.4f}\n")
# -------------------------
# Step 6: Plot scatter plot of actual data and predicted regression line
# -------------------------
plt.scatter(x, y, color='red', label='Actual data')
plt.plot(x, y_pred, color='blue', label='Predicted line')
plt.xlabel('Study time (hours)')
plt.ylabel('Score out of 10')
plt.title('Study Time vs Score with Linear Regression')
plt.legend()
plt.show()
# -------------------------
# Step 7: Compare coefficients from both methods
# -------------------------
print("Comparison of coefficients:")
print(f" Pedhazur method: B0 = {B0_pedhazur:.4f}, B1 = {B1_pedhazur:.4f}")

4

print(f" Calculus method: B0 = {B0_calc:.4f}, B1 = {B1_calc:.4f}")
print("They are essentially the same, confirming correctness.\n")
# -------------------------
# Step 8: Predict score for study time = 10 hours
# -------------------------
study_time = 10
predicted_score = B0_pedhazur + B1_pedhazur * study_time
print(f"Predicted score for {study_time} hours of study: {predicted_score:.2f}")