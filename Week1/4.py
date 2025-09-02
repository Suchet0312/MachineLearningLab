import numpy as np

# Create a 2D array (for example: 4x6)
arr = np.array([[1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23, 24]])

# Extract first four columns
first_four_cols = arr[:, :4]

print(first_four_cols)
