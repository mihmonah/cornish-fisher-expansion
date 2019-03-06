from math import sqrt
import numpy as np
from scipy import stats


def generate_emp_dist(n: int, df: float, mu: float, sizes: np.array = None,
                      distr='chi2'):
    """
        Function for generating array of means of random variates with empirical distribution,
        :param n:
        :param mu:
        :param sizes:
        :param df: a shape parameter (if it exists)
        :param distr:
    """
    distribution = getattr(stats, distr)
    vector = np.zeros(n, dtype=float)
    for i in range(n):
        vector[i] = np.mean(distribution.rvs(df=df, size=sizes[i])) - mu
    return vector


def obtain_moments(distr: str, df: float = 0):
    if distr == 'chi2':
        mu = df
        sigma = (2 * df) ** (- 1 / 2)
        mu3 = sqrt(8 / df)
        mu4 = 12 / df
        return mu, sigma, mu3, mu4
