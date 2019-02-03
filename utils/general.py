import numpy as np
from scipy import stats


def generate_chi_squared_dist(n: int, df: float, mu: float, sizes: np.array = None):
    """
        Function for generating array of means of random variates with chi-squared distribution,
        shifted by mu
    """
    chi_vector = np.zeros(n, dtype=float)
    for i in range(n):
        chi_vector[i] = np.mean(stats.chi2.rvs(df=df, size=sizes[i])) - mu
    return chi_vector
