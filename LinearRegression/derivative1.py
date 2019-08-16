import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-8, 5, 400)
a = 4
b = 9
c = -20
y = a * x * x + b * x + c
ydev = 2 * a * x + b

fig = plt.figure()
fig.subplots_adjust(hspace=.5)
ax1 = fig.add_subplot(111)

ax1.plot(x, y, x, ydev)
ax1.set(title=f"y = {a}x^2 + {b}x + {c}")
plt.grid(True)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

plt.show()
