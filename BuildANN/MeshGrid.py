import numpy as np

x= np.array(np.meshgrid([1, 2, 3, 4], [5, 6, 7, 8])).T.reshape(-1, 2)
print(x.shape)
print(x)

a = np.linspace(0, 1, 5)
#y = np.array(np.meshgrid(a, a)).T.reshape(-1, 2)
#print(y)

xx, yy = np.meshgrid(a, a)

print(xx)
