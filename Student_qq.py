import argparse
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plotting_utils.qq_plotting import qqprobplot
from src.settings import DistributionError, supported_distributions
from utils.general import generate_emp_dist, obtain_moments
from utils.student import calculate_student_qq_approximation


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

R = 2  # R (parameter r from formulas)
x_vec = np.linspace(-10, 10, POINTS)
MU, SIGMA, MU3, MU4 = obtain_moments(emp_distribution, emp_df)
probabilities = np.linspace(0, 1, POINTS)

# generation of sample size:
Nn = stats.nbinom.rvs(n=R, p=1 / NN_NUMBER, size=POINTS)

# generation of random value and calculation of statistics' values
chi_vector = generate_emp_dist(n=POINTS, df=emp_df, mu=MU, sizes=Nn)
TNn_emp = chi_vector * SIGMA * sqrt(R * (NN_NUMBER - 1) + 1)
TNn_emp.sort(axis=0)

# quantiles
q_emp = TNn_emp
q_appr = calculate_student_qq_approximation(
    probs=probabilities, r=R, nn=NN_NUMBER, mu3=MU3, mu4=MU4)

# plotting:
print('Start plotting')
plt.interactive(False)
plt.ylim(-5, 5)
plt.xlim(-5, 5)
qqprobplot(q_emp, q_appr, plot=plt)

plt.title(f"QQ-plot for Student case: {NN_NUMBER} variables with" + r" $\chi^2(1)$")

plt.legend()
plt.show()
print('Finish plotting')
