from math import sqrt

from scipy import stats


def calculate_student_approximation(xv, r: float = 1, order: int = 2, nn: int = 10,
                                    mu3: float = 1, mu4: float = 1):
    """
        Function for calculating Chebyshev-Edgeworth approximation for Student case

        :param xv: points for approximation
        :param r: parameter of Student distribution
        :param nn: index of Nn
        :param order: order of approximation (first or second)
        :param mu3: skewness of X
        :param mu4: kurtosis of X
        :returns: vector of Student approximation, or cdf, if order is 0
    """
    cdf = stats.t.cdf(x=xv, df=2 * r)
    if order == 0:
        return cdf

    pdf = stats.t.pdf(x=xv, df=2 * r)
    gn = r * (nn - 1) - 1

    a_1 = (mu3 * ((r - 1) * xv ** 2 - r)) / (3 * (2 * r - 1))
    a_21 = (xv / (36 * (2 * r - 1)))\
        * ((2 * mu3 ** 2 * ((r - 2) * (r - 3) * xv ** 4 + 10 * r * (2 - r) * xv ** 2
                            + 15 * r ** 2)) / (2 * r + xv ** 2))
    a_22 = (xv / (36 * (2 * r - 1))) * (3 * mu4 * ((r - 2) * xv ** 2 - 3 * r)
                                        + 9 * (r - 2) * (xv ** 2 + 1))

    if order == 1:
        return cdf + a_1 * pdf / sqrt(gn)
    elif order == 2:
        return cdf - a_1 * pdf / sqrt(gn) - pdf * (a_21 + a_22) / gn


def calculate_student_qq_approximation(probs, r: float = 1, nn: int = 10,
                                       mu3: float = 1, mu4: float = 1):
    """
        Function for calculating Cornish-Fisher approximation for Student case

        :param probs: vector of probabilities
        :param r: parameter of Student distribution
        :param nn: index of Nn
        :param mu3: skewness of X
        :param mu4: kurtosis of X
        :returns: vector of Student quantiles approximation
    """
    q_s = stats.t.ppf(q=probs, df=2 * r)
    gn = r * (nn - 1) + 1
    b_0 = (r - 1) * q_s ** 2 - r
    b_1 = (mu3 * b_0) / (3 * (2 * r - 1))
    b_21 = - ((2 * r + 1) * q_s * (mu3 ** 2) * b_0 ** 2)\
        / (2 * (2 * r + q_s ** 2) * 9 * ((2 * r - 1) ** 2))
    b_22 = (2 * (mu3 ** 2) * (r - 1) * q_s * b_0) / (9 * (2 * r - 1) ** 2)
    b_23 = (q_s / (36 * (2 * r - 1))) \
        * (2 * mu3 ** 2 * ((r - 2) * (r - 3) * q_s ** 4 + 10 * r * (2 - r) * q_s ** 2 + 15 * r ** 2)
           / (2 * r + q_s ** 2) + 3 * mu4 * ((r - 2) * q_s ** 2 - 3 * r)
           + 9 * (r - 2) * (q_s ** 2 + 1))
    b_2 = b_21 + b_22 + b_23
    return q_s + b_1 / sqrt(gn) + b_2 / gn
