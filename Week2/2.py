import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
y = x
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
ay = fig.add_axes([0.2,0.5,0.2,0.2])
ay.plot(x,y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ay.set_xlabel('x')
ay.set_ylabel('y')
ax.set_title('title')
ay.set_title('title2')
plt.show()