supported_sitributions = {
    'chi2': '$\chi^2$',
    'gamma': '$\Gamma$',
    'norm': '$\mathcal{N}$'}


class DistributionError(Exception):
    def __init__(self, code):
        self.code = code

    def __str__(self):
        exception_msg = "Distribution {} is not supported".format(self.code)
        return exception_msg
