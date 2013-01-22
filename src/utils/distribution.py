import operator
import math
from itertools import combinations
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import Orange

class JointDistributions():
    """
    Class for representing and approximating joint distributions of attributes
    over a data set.
    """

    def __init__(self, data, kirkwood_level=3):
        self.kirkwood_level = kirkwood_level
        self.data = data
        self.domain_dist = Orange.statistics.distribution.Domain(data)
        self.interpolators = {}

    def density(self, attrs_vals):
        """
        Returns approximated probability density function value for the given
        set of attributes values. Returned values are not normalized.
        
        Args:
            attrs_vals: a dictionary of attributes names or indices and
                        the corresponding attributes values.
        Returns:
            density function value.
        """
        return self._density(attrs_vals.keys(), attrs_vals.values())

    def _density(self, attrs, vals):
        attrs = tuple(attrs)
        vals = tuple(vals)
        if len(attrs) < self.kirkwood_level:
            return self._interpolated_density(attrs, vals)
        k = self._kirkwood_approx(attrs, vals)
        return k if not math.isnan(k) else 0.0

    def _interpolated_density(self, attrs, vals):
        if len(attrs) == 1:
            try:
                return self.domain_dist[attrs[0]].density(vals[0])
            except AttributeError:
                return self.domain_dist[attrs[0]][vals[0]]
        if not attrs in self.interpolators:
            freqs = {}
            for d in self.data:
                key = tuple(float(d[a]) for a in attrs)
                if key not in freqs:
                    freqs[key] = 0
                freqs[key] += 1
            a = np.array(freqs.keys())
            v = np.array(freqs.values())
            self.interpolators[attrs] = LinearNDInterpolator(a, v, 0.0)
        return self.interpolators[attrs]([vals])[0]
    
    def _kirkwood_approx(self, attrs, vals):
        return reduce(operator.div,
                (reduce(operator.mul,
                    (self._density((attrs[i] for i in indices),
                        (vals[i] for i in indices))
                        for indices in combinations(range(len(attrs)), n)))
                    for n in xrange(len(attrs)-1, 0, -1)))


