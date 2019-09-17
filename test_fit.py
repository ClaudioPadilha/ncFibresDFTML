import numpy as np, matplotlib.pyplot as plt, re
from numpy import pi, r_
from scipy import optimize
from matplotlib import rcParams
rcParams['figure.figsize'] = (6, 4)
rcParams['legend.fontsize'] = 16
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 20

# Data points: x = distance, y = formation energy
# x = np.array([[7,  12,  10,   3,   8],
# 			[6,  10,   8,   2,   6],
# 			[8,  16,  12,   4,   8],
# 			[12,  25,  20,   8,  15],
# 			[1,   4,   2,   1,   0],
# 			[6,  13,  10,   4,   7],
# 			[6,  10,   8,   2,   6],
# 			[4,   8,   6,   2,   4],
# 			[10,  20,  16,   6,  12],
# 			[8,  15,  12,   4,   9],
# 			[5,   8,   6,   1,   4],
# 			[7,  16,  12,   5,   8]], dtype=float)
x = np.array([[10,  12,  10,   3,   8],
			[9,  10,   8,   2,   6],
			[13,  16,  12,   4,   8],
			[18,  25,  20,   8,  15],
			[4,   4,   2,   1,   0],
			[10,  13,  10,   4,   7],
			[9,  10,   8,   2,   6],
			[7,   8,   6,   2,   4],
			[15,  20,  16,   6,  12],
			[12,  15,  12,   4,   9],
			[8,   8,   6,   1,   4],
			[12,  16,  12,   5,   8]], dtype=float)
# x = np.array([[10,  7,  10,   3,   8],
# 			[9,  6,   8,   2,   6],
# 			[13,  8,  12,   4,   8],
# 			[18,  12,  20,   8,  15],
# 			[4,   1,   2,   1,   0],
# 			[10,  6,  10,   4,   7],
# 			[9,  6,   8,   2,   6],
# 			[7,   4,   6,   2,   4],
# 			[15,  10,  16,   6,  12],
# 			[12,  8,  12,   4,   9],
# 			[8,   5,   6,   1,   4],
# 			[12,  7,  12,   5,   8]], dtype=float)
# x = np.array([[10,  7,  12,   3,   8],
# 			[9,  6,   10,   2,   6],
# 			[13,  8,  16,   4,   8],
# 			[18,  12,  25,   8,  15],
# 			[4,   1,   4,   1,   0],
# 			[10,  6,  13,   4,   7],
# 			[9,  6,   10,   2,   6],
# 			[7,   4,   8,   2,   4],
# 			[15,  10,  20,   6,  12],
# 			[12,  8,  15,   4,   9],
# 			[8,   5,   8,   1,   4],
# 			[12,  7,  16,   5,   8]], dtype=float)
y = np.array([-18.30735, -16.08481, -23.96499, -36.04686,
			-4.56364, -18.19864, -15.88926, -11.46830, 
			-29.20223, -22.81719, -13.58684, -21.72626], dtype=float)

# Fit the first set
fitfunc = lambda p, x: [p[0]*i[0] + p[1]*i[1] + p[2]*i[2] + p[3]*i[3] + p[4]*i[4] + p[5] for i in x] # Target function
errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function


p0 = [0., 0., 0., 0., 0., 0.] # Initial guess for the parameters

# print(errfunc(p0, x, y))
# print(fitfunc(p0, x[1]))


p1, success = optimize.leastsq(errfunc, p0[:], args=(x, y))

n = np.sum([i ** 2 for i in p1])
print(success, p1/n)

for i, j in zip(fitfunc(p1, x), y):
	print('{:9.5f}\t{:9.5f}'.format(j,i))


# xf = np.linspace(2, 14, 500)
# plt.plot(x, y, "bo", xf, fitfunc(p1, xf), "r--") # Plot of the data and the fit

# # Legend the plot
# # plt.title("Oscillations in the compressed trap")
# plt.xlabel(r"Layer tickness (\AA)")
# plt.ylabel(r"Formation energy (eV / \AA$^2$)")
# # plt.legend(('data', 'fit'))
# plt.ylim([0.025, 0.15])
# plt.xlim([2, 12])
# ax = plt.axes()

# plt.text(0.7, 0.7,
#          'E$_0$ : {:3.2f}\n E$_1$ : {:3.2f}\n $\lambda$ : {:3.2f}'.format(p1[0], p1[1], p1[2]),
#          fontsize=18,
#          horizontalalignment='left',
#          verticalalignment='center',
#          transform=ax.transAxes)
# plt.tight_layout()

# # plt.savefig("fit_deltae_x_d.jpg", format="jpg")
# plt.show()