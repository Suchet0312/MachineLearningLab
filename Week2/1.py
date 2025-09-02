import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = x
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('title')
fig.show()

