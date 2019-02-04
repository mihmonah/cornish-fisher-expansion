from math import sqrt

import numpy as np
# import rpy2.robjects as robjects
from scipy import stats

GENERATION_SIZE = 10000


def calculate_laplace_approximation(xv, loc: float, scale: float, s: float = 2, nn: int = 10,
                                    mu3: float = 1, mu4: float = 1, order: int = 2):
    """

    :param xv: points for approximation
    :param loc: location parameter of Laplace distribution
    :param scale: scale parameter of Laplace distribution
    :param s: parameter of Laplace distribution
    :param nn: index of Nn
    :param order: order of approximation (first or second)
    :param mu3: skewness of X
    :param mu4: kurtosis of X
    :returns: vector of Laplace approximation, or cdf, if order is 0
    """
    cdf = stats.laplace.cdf(x=xv, loc=loc, scale=scale / (sqrt(2 * s)))
    if order == 0:
        return cdf

    pdf = stats.laplace.pdf(x=xv, loc=loc, scale=scale / (sqrt(2 * s)))

    a_1 = (mu3 / 6) * (- np.power(xv, 2) + np.abs(xv) / sqrt(2 * s) + 1 / (2 * s))
    a_21 = (xv * mu4 / (48 * s)) * (3 - 2 * s * np.power(xv, 2) + 3 * sqrt(2 * s) * np.abs(xv)) \
        + (xv * mu3 ** 2 / (144 * s)) * (20 * s * xv * xv - (np.abs(xv) * np.power(xv, 2) * 3)
                                            * (2 * s) ** (3 / 2) - 15 * sqrt(2 * s) * abs(xv)
                                                                 - 15)
    a_22 = (xv * (1 - s) / (8 * s)) * (sqrt(2 * s) * np.abs(xv) + 1)

    if order == 1:
        return cdf + pdf * (a_1 / sqrt(nn))
    elif order == 2:
        return cdf + pdf * (a_1 / sqrt(nn)) + pdf * ((a_21 + a_22) / nn)


def calculate_laplace_qq_approximation(probs, loc: float, scale: float, s: float = 2, nn: int = 10,
                                       mu3: float = 1, mu4: float = 1):
    """

    :param probs: vector of probabilities
    :param loc: location parameter of Laplace distribution
    :param scale: scale parameter of Laplace distribution
    :param s: parameter of Laplace distribution
    :param nn: index of Nn
    :param mu3: skewness of X
    :param mu4: kurtosis of X
    :returns: vector of Laplace quantiles approximation
    """
    q_l = stats.laplace.ppf(q=probs, loc=loc, scale=scale / (sqrt(2 * s)))
    a_00 = np.abs(q_l) / sqrt(2 * s) + 1 / (2 * s) - q_l ** 2
    a_01 = (np.sign(q_l)) * sqrt(2 * s) * a_00 ** 2
    a_02 = (np.sign(q_l) / sqrt(2 * s) - 2 * q_l) * a_00
    b_01 = ((mu3 ** 2) / 36) * (a_01 - a_02)
    b_02 = ((q_l * mu3 ** 2) / (144 * s)) * (20 * s * q_l ** 2 - (2 * s) ** (3 / 2)
                                             * (np.abs(q_l)) ** 3 - 15 * sqrt(2 * s) * np.abs(q_l)
                                             - 15)
    b_03 = ((q_l * mu4) / (48 * s)) * (3 - 2 * s * q_l ** 2 + 3 * sqrt(2 * s) * np.abs(q_l))
    b_04 = (q_l * (1 - s) / (8 * s)) * (sqrt(2 * s) * np.abs(q_l) + 1)
    b_2 = b_01 + b_02 + b_03 + b_04
    return q_l - (mu3 / (6 * sqrt(nn))) * a_00 + b_2 / nn


def generate_max_discrete_pareto_dist(shape: float, rv_num: int, n: int, m: int = GENERATION_SIZE):
    """
        Function for generating random variables sample with max of discrete Pareto distribution

        :parameter shape: shape parameter of Pareto distribution
        :parameter rv_num: number of random variables
        :parameter n: size of sample to return
        :parameter m: size of interval for generation (m >> 1)
        :returns: array of int
    """
    values = np.array(range(1, m + 1), int)

    probs = np.zeros(m, dtype=float)
    for i in range(m - 1):
        j = i + 1
        probs[i] = (j / (shape + j)) ** rv_num - ((j - 1) / (shape + j - 1)) ** rv_num
    probs[m - 1] = 1 - np.sum(probs)

    return np.random.choice(a=values, size=n, p=probs)


# def rgenerate_max_discrete_pareto_dist(shape: float, rv_num: int, n: int, m: int = GENERATION_SIZE):
#     """
#         R function for generating of sample with max of discrete Pareto distribution
#
#         :parameter shape: shape parameter of Pareto distribution
#         :parameter rv_num: number of random variables
#         :parameter n: size of sample to return
#         :parameter m: size of choose interval (m >> 1)
#         :returns: array of int
#     """
#     robjects.r("""
#         # R function for generating of sample with max of discrete Pareto distribution
#         f <- function(sn, n, k, m) {
#             prob_int <- rep(0,k)
#             Nn <- c(1:k)
#             my_prob <- c(1:k)
#             for(j in 1:m){
#                 my_prob[j]=(j/(sn+j))^n-((j-1)/(sn+j-1))^n
#             }
#             Uni_probs <- runif(k)
#             prob_int <- my_prob
#             for(j in 2:m){
#                 prob_int[j]=prob_int[j]+prob_int[j-1]
#             }
#             for (i in 1:k){
#                 Nn[i] <- findInterval(Uni_probs[i],prob_int)
#             }
#             Nn
#         }
#     """)
#     r_f = robjects.r['f']
#     return r_f(shape, rv_num, n, m)
