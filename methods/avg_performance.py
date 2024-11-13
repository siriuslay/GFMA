
import numpy as np


a = np.array()
b = np.array()
c = np.array()
d = np.array()
e = np.array()
f = np.array()

results = (a + b + c + d + e + f) / 6
# c = np.array([55.55555555555556, 55.55555555555556, 26.666666666666668, 55.55555555555556, 55.55555555555556, 26.666666666666668, 55.55555555555556, 55.55555555555556, 55.55555555555556, 55.55555555555556])
# print(c)

print(f'{round(np.mean(results), 2)}±{round(np.std(results, ddof=0), 2)}')