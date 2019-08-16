import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 400)

fig = plt.figure()
fig.subplots_adjust(hspace=.5)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


ax1.plot(x, np.sin(x), x, np.cos(x))
ax1.set(title="sin(x)")

ax2.plot(x, x ** 2, x, 2*x)
ax2.set(title="x^2")

ax3.plot(x, x ** 3, x, 3 * x**2)
ax3.set(title="x^3")

ax4.set(title="log(x)")
z = np.linspace(0, 2, 100)
ax4.plot(z, np.log(z), z, 1/z)

plt.show()
