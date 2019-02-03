from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from utils.general import generate_chi_squared_dist
from utils.laplace import calculate_laplace_approximation, generate_max_discrete_pareto_dist

POINTS = 10000  # number of points for plotting
NN_NUMBER = 10  # number of Nn
sn = 2  # parameter of Nn distribution
XI = 1
MU = 1
SIGMA = (2 * XI) ** (- 1 / 2)
MU3 = sqrt(8 / XI)
MU4 = 12 / XI
x_vec = np.array(range(POINTS), float)
x_vec = (x_vec - POINTS / 2) / 100
emp_vec = np.linspace(0, 1, POINTS)


array_of_vals = generate_max_discrete_pareto_dist(shape=sn, rv_num=NN_NUMBER, n=POINTS)

chivec = generate_chi_squared_dist(n=POINTS, df=XI, mu=MU, sizes=array_of_vals) * SIGMA\
         * sqrt(NN_NUMBER)
TNn_emp = sorted(chivec)

laplace_cdf = calculate_laplace_approximation(
    xv=x_vec, loc=0, scale=1, order=0)
TNn_appr1 = calculate_laplace_approximation(
    xv=x_vec, loc=0, scale=1, s=sn, nn=NN_NUMBER, mu3=MU3, mu4=MU4, order=1)
TNn_appr = calculate_laplace_approximation(
    xv=x_vec, loc=0, scale=1, s=sn, nn=NN_NUMBER, mu3=MU3, mu4=MU4, order=2)


print('Start plotting')
plt.interactive(False)
plt.ylim(0, 1)
plt.xlim(-3, 3)

plt.plot(TNn_emp, emp_vec, label='empirical')
plt.plot(x_vec, TNn_appr1, label='1st approximation')
plt.plot(x_vec, TNn_appr, label='2nd approximation')
plt.plot(x_vec, laplace_cdf, label='Pure Laplace')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title(r"Laplace case: Distribution - $\chi^2(1)$" + f", {NN_NUMBER} variables")

plt.legend()
plt.show()
print('Finish plotting')
