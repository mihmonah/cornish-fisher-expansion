import argparse
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from plotting_utils.qq_plotting import qqprobplot
from src.settings import DistributionError, supported_distributions
from utils.general import generate_emp_dist, obtain_moments
from utils.laplace import calculate_laplace_qq_approximation, generate_max_discrete_pareto_dist


parser = argparse.ArgumentParser(description='Process some strings')
parser.add_argument('--distr', default='chi2', help='Empirical distribution')
parser.add_argument('--df', default=1, help='Shape parameter for empirical distribution')
parser.add_argument('--N', default=10, help='Number of observations')
parser.add_argument('--points', default=10000, help='Number of points for plotting')

emp_distribution = parser.parse_args().distr
emp_df = parser.parse_args().df
NN_NUMBER = parser.parse_args().N
POINTS = parser.parse_args().points

if emp_distribution not in supported_distributions:
    raise DistributionError(emp_distribution)

GENERATION_SIZE = 10000  # parameter for initialization of Nn_s(x=m), m in N
sn = 2  # parameter of Nn distribution
MU, SIGMA, MU3, MU4 = obtain_moments(emp_distribution, emp_df)
probabilities = np.linspace(0, 1, POINTS)


array_of_vals = generate_max_discrete_pareto_dist(shape=sn, rv_num=NN_NUMBER, n=POINTS)
chivec = generate_emp_dist(n=POINTS, df=emp_df, mu=MU, sizes=array_of_vals) * SIGMA \
         * sqrt(NN_NUMBER)

q_emp = sorted(chivec)
q_appr = calculate_laplace_qq_approximation(
    probs=probabilities, loc=0, scale=1, s=sn, nn=NN_NUMBER, mu3=MU3, mu4=MU4)


# plotting:
print('Start plotting')
plt.interactive(False)

plt.ylim(-4, 4)
plt.xlim(-3, 3)
plt.axhline(linewidth=0.5, color='black')
plt.axvline(linewidth=0.5, color='black')
qqprobplot(q_emp, q_appr, plot=plt)

plt.title(f"QQ-plot for Laplace case: {NN_NUMBER} variables with" + r" $\chi^2(1)$")

plt.legend()
plt.show()
print('Finish plotting')
