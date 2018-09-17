# coding=utf-8
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

xs = range(500)
ys = randn(500)*1. + 10.
plt.plot(xs, ys)

print('Mean of readings is {:.3f}'.format(np.mean(ys)))
plt.show()


