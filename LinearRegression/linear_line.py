import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 200)
a = 0.2
b = 2
y = a * x + b

plt.plot(x, y)
plt.grid(True)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.xlabel('x')
plt.ylabel(f'y={a}*x + {b}')
plt.show()