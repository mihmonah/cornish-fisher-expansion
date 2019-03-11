from math import sqrt
import numpy as np
from scipy import stats


def generate_emp_dist(n: int, df: float, mu: float, sizes: np.array = None,
                      distr='chi2'):
    """
        Function for generating array of means of random variates with empirical distribution,
        shifted by `mu`
        Current version supports only Chi-squared distribution

        :param n: size of returning vector
        :param mu: shifting parameter
        :param sizes: array of sizes for each mean
        :param df: a shape parameter (if it exists)
        :param distr: empirical distribution, only Chi-squared is supported
        :returns vector: vector of means of random variates with empirical distribution
    """
    distribution = getattr(stats, distr)
    vector = np.zeros(n, dtype=float)
    for i in range(n):
        vector[i] = np.mean(distribution.rvs(df=df, size=sizes[i])) - mu
    return vector


def obtain_moments(distr: str, df: float = 0):
    """
        Function for obtaining first 4 moments of random variable
        Current version supports only Chi-squared distribution

        :param distr: distribution, only Chi-squared is supported
        :param df: a shape parameter
        :return (mu, sigma, mu3, mu4): tuple with first 4 moments
    """
    if distr == 'chi2':
        mu = df
        sigma = (2 * df) ** (- 1 / 2)
        mu3 = sqrt(8 / df)
        mu4 = 12 / df
        return mu, sigma, mu3, mu4
