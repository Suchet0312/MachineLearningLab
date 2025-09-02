import numpy as np

# Two sample arrays
arr1 = np.array([3, 5, 7, 2, 9])
arr2 = np.array([1, 6, 4, 2, 10])

# Find positions where arr1 >= arr2
positions = np.where(arr1 >= arr2)

print("Positions:", positions[0])   # positions as a 1D array
