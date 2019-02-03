from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from plotting_utils.qq_plotting import qqprobplot
from utils.general import generate_chi_squared_dist
from utils.laplace import calculate_laplace_qq_approximation, generate_max_discrete_pareto_dist

POINTS = 1000  # number of points for plotting
NN_NUMBER = 10  # number of Nn
GENERATION_SIZE = 10000  # parameter for initialization of Nn_s(x=m), m in N
sn = 2  # parameter of Nn distribution
XI = 1
MU = 1
SIGMA = (2 * XI) ** (- 1 / 2)
MU3 = sqrt(8 / XI)
MU4 = 12 / XI
probabilities = np.linspace(0, 1, POINTS)


array_of_vals = generate_max_discrete_pareto_dist(shape=sn, rv_num=NN_NUMBER, n=POINTS)
chivec = generate_chi_squared_dist(n=POINTS, df=XI, mu=MU, sizes=array_of_vals) * SIGMA\
         * sqrt(NN_NUMBER)

q_emp = sorted(chivec)
q_appr = calculate_laplace_qq_approximation(
    probs=probabilities, loc=0, scale=1, s=sn, nn=NN_NUMBER, mu3=MU3, mu4=MU4)


# plotting:
print('Start plotting')
plt.interactive(False)
plt.ylim(-4, 4)
plt.xlim(-3, 3)
qqprobplot(q_emp, q_appr, plot=plt)

plt.title(f"QQ-plot for Laplace case: {NN_NUMBER} variables with" + r" $\chi^2(1)$")

plt.legend()
plt.show()
print('Finish plotting')
