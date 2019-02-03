from math import sqrt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from utils.general import generate_chi_squared_dist
from utils.student import calculate_student_approximation


POINTS = 10000
R = 1  # R (parameter r from formulas)
NN_NUMBER = 10  # OBSERVATIONS (number of observations)
# var
XI = 1
MU = 1
SIGMA = (2 * XI) ** (- 1 / 2)
MU3 = sqrt(8 / XI)
MU4 = 12 / XI
x_vec = np.linspace(-10, 10, POINTS)

# generation of sample size:
Nn = stats.nbinom.rvs(n=R, p=1 / NN_NUMBER, size=POINTS)

chi_vector = generate_chi_squared_dist(n=POINTS, df=XI, mu=MU, sizes=Nn)
TNn_emp = chi_vector * SIGMA * sqrt(R * (NN_NUMBER - 1) + 1)
TNn_emp.sort(axis=0)

student_cdf = calculate_student_approximation(xv=x_vec, r=R, order=0)
TNn_apr1 = calculate_student_approximation(xv=x_vec, r=R, order=1, nn=NN_NUMBER, mu3=MU3, mu4=MU4)
TNn_apr2 = calculate_student_approximation(xv=x_vec, r=R, order=2, nn=NN_NUMBER, mu3=MU3, mu4=MU4)
emp_vec = np.linspace(0, 1, len(TNn_emp))

print('Start plotting')
plt.interactive(False)
plt.ylim(0, 1)
plt.xlim(-3, 3)
plt.plot(TNn_emp, emp_vec, label='empirical')
plt.plot(x_vec, TNn_apr1, label='1st approximation')
plt.plot(x_vec, TNn_apr2, label='2nd approximation')
plt.plot(x_vec, student_cdf, label='Pure Student')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title(r"Student case: Distribution - $\chi^2(1)$, n = 10, s = 1")

plt.legend()
plt.show()
print('Finish plotting')
