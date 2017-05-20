import numpy as np
import copy


a = np.zeros((2,2))
b = a.reshape((4,1)).copy()
b[2][0] = 1
print(a)
print(b)