import argparse
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from src.settings import DistributionError, supported_distributions
from utils.general import generate_emp_dist, obtain_moments
from utils.laplace import calculate_laplace_approximation, generate_max_discrete_pareto_dist

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

sn = 2  # parameter of Nn distribution
MU, SIGMA, MU3, MU4 = obtain_moments(emp_distribution, emp_df)
x_vec = np.linspace(-10, 10, POINTS)
emp_vec = np.linspace(0, 1, POINTS)


array_of_vals = generate_max_discrete_pareto_dist(shape=sn, rv_num=NN_NUMBER, n=POINTS)

chivec = generate_emp_dist(n=POINTS, df=emp_df, mu=MU, sizes=array_of_vals) * SIGMA \
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

plt.title(r"Laplace case: Distribution - {}, n = {}, s = 1".format(
    supported_distributions[emp_distribution],
    NN_NUMBER)
)
plt.legend()
plt.show()
print('Finish plotting')
