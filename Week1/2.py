import numpy as np

Assign_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

Assign_1[Assign_1 % 2 != 0] = -1

print(Assign_1.reshape(3, 3))
