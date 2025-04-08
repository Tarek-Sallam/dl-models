from autodiff.Tensor import Tensor
import numpy as np
import matplotlib.pyplot as plt


num = 10
x = 10*np.random.rand(100)
y = 10*np.random.rand(100)
features = np.array([x, y])
print(features)
labels = [0 if y - x - 5*(np.random.rand(1) - 0.5) > 0 else 1 for x, y in zip(x, y)]
colors = ['red' if label == 0 else 'blue' for label in labels]
fig, ax = plt.subplots()
ax.scatter(x, y, c=colors)
plt.show()

