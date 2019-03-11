import numpy as np
from scipy import stats
from scipy.stats.morestats import (
    _add_axis_labels_title,
    amax,
    amin
)


def qqprobplot(x, y, fit=True, plot=None, rvalue=False):
    """
        Modified `probplot` method from `scipy.stats`
        Takes two arrays of quantiles for a probability plot, and optionally show the plot.
        Generates a probability plot of sample data against the empirical quantiles
        `probplot` optionally calculates a best-fit line for the data and plots the
        results using Matplotlib or a given plot function.

        :param x: sample/response data from which `qqprobplot` creates the plot
        :param y: empirical Sample/response data from which `qqprobplot` creates the plot
        :param fit: fit a least-squares regression (best-fit) line to the sample data if True
        :param plot: if given, plots the quantiles and least squares fit
        `plot` is an object that has to have methods "plot" and "text".


        :returns (osm, osr): tuple of theoretical quantiles (osm, or order statistic medians) and
        empirical quantiles (osr).  `osr` is simply sorted input `x`.
        :returns (slope, intercept, r): tuple  containing the result of the least-squares fit, if
        that is performed by `probplot`. `r` is the square root of the coefficient of
        determination.

    """
    x = np.asarray(x)
    _perform_fit = fit or (plot is not None)
    if x.size == 0:
        if _perform_fit:
            return (x, x), (np.nan, np.nan, 0.0)
        else:
            return x, x

    osm = sorted(y)
    osr = sorted(x)
    if _perform_fit:
        # perform a linear least squares fit.
        slope, intercept, r, prob, sterrest = stats.linregress(osm, osr)

    if plot is not None:
        plot.plot(osm, osr, 'bo', markersize=2)
        plot.plot(np.arange(-10, 10), np.arange(-10, 10), linewidth=0.5, color='black')
        plot.axhline(linewidth=0.5, color='black')
        plot.axvline(linewidth=0.5, color='black')
        _add_axis_labels_title(plot, xlabel='Theoretical quantiles',
                               ylabel='Approximate quantiles',
                               title='Probability Plot')

        # Add R^2 value to the plot as text
        if rvalue:
            xmin = amin(osm)
            xmax = amax(osm)
            ymin = amin(x)
            ymax = amax(x)
            posx = xmin + 0.70 * (xmax - xmin)
            posy = ymin + 0.01 * (ymax - ymin)
            plot.text(posx, posy, "$R^2=%1.4f$" % r**2)

    if fit:
        return (osm, osr), (slope, intercept, r)
    else:
        return osm, osr
