import operator
import math
from itertools import combinations
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import Orange

class JointDistributions():
    """
    Class for representing and approximating joint distributions of attributes
    over a data set.
    """

    def __init__(self, data, kirkwood_level=3):
        """
        Constructs an instance of JointDistibutions from the given data set.

        Args:
            data: data set, instance of Orange.data.Table,
            kirkwood_level: minimum order of joint to apply Kirkwood approximation.
        """
        self.kirkwood_level = kirkwood_level
        self.data = data
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
            r = self._interpolated_density(attrs, vals)
        else:
            k = self._kirkwood_approx(attrs, vals)
            r = k if not (math.isnan(k) or math.isinf(k)) else 0.0
        #print attrs, vals, r
        return r

    def _interpolated_density(self, attrs, vals):
        if not attrs in self.interpolators:
            freqs = {}
            for d in self.data:
                key = tuple(float(d[a]) for a in attrs)
                if key not in freqs:
                    freqs[key] = 0
                freqs[key] += 1
            v = np.array(freqs.values())
            if len(attrs) == 1:
                a = np.array(sorted(k[0] for k in freqs.keys()))
                # TODO: Error in interpolation
                interp_fun = interp1d(a, v, bounds_error=False, fill_value=0.0)
                self.interpolators[attrs] = lambda x: interp_fun(x)[0]
            else:
                a = np.array(freqs.keys())
                self.interpolators[attrs] = LinearNDInterpolator(a, v, 0.0)
        return self.interpolators[attrs](vals)
    
    def _kirkwood_approx(self, attrs, vals):
        return reduce(operator.div,
                (reduce(operator.mul,
                    (self._density((attrs[i] for i in indices),
                        (vals[i] for i in indices))
                        for indices in combinations(range(len(attrs)), n)))
                    for n in xrange(len(attrs)-1, 0, -1)))
