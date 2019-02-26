from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plotting_utils.qq_plotting import qqprobplot
from utils.general import generate_emp_dist
from utils.student import calculate_student_qq_approximation


POINTS = 10000
R = 2  # R (parameter r from formulas)
NN_NUMBER = 10  # OBSERVATIONS (number of observations)
# var
XI = 1
MU = 1
SIGMA = (2 * XI) ** (- 1 / 2)
MU3 = sqrt(8 / XI)
MU4 = 12 / XI
probabilities = np.linspace(0, 1, POINTS)

# generation of sample size:
Nn = stats.nbinom.rvs(n=R, p=1 / NN_NUMBER, size=POINTS)

# generation of random value and calculation of statistics' values
chi_vector = generate_emp_dist(n=POINTS, df=XI, mu=MU, sizes=Nn)
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
