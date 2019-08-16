import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 1, 0.01)
y = -np.log(1-x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.grid()
plt.title('-log()',fontsize=10)
plt.show()