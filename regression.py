import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

labels = []
desc = []
targets = []
with open("regression_data.dat") as file:
	lines = file.readlines()

# print(lines)



for line in lines:
	linestrip = line.strip().split(',')
	labels.append(linestrip[0])
	desc.append(linestrip[1:-1])
	targets.append(linestrip[-1])

# print(labels)

# print(targets)

# print(desc)

y = np.array(targets, dtype=float)
X = np.array(desc, dtype=int)

# print(X)
# print(y)

reg = LinearRegression().fit(X, y)
print(reg.score(X, y))

print(reg.coef_)
print(reg.intercept_)
