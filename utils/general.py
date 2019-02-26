import numpy as np
from scipy import stats


def generate_emp_dist(n: int, df: float, mu: float, sizes: np.array = None,
                      distr='chi2'):
    """
        Function for generating array of means of random variates with chi-squared distribution,
        shifted by mu
        :param distr:
    """
    distribution = getattr(stats, distr)
    vector = np.zeros(n, dtype=float)
    for i in range(n):
        vector[i] = np.mean(distribution.rvs(df=df, size=sizes[i])) - mu
    return vector
