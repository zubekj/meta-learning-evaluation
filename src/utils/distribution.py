import operator
import math
from itertools import combinations, product
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d, NearestNDInterpolator
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
        self.freqs = {}

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
            r = self._freqs_density(attrs, vals)
        else:
            k = self._kirkwood_approx(attrs, vals)
            r = k if not (math.isnan(k) or math.isinf(k)) else 0.0
        return r

    def _freqs_density(self, attrs, vals):
        if not attrs in self.freqs:
            fq = {}
            for d in self.data:
                key = tuple(round(float(d[a]), 4) for a in attrs)
                if key not in fq:
                    fq[key] = 0
                fq[key] += 1
            self._build_interpolator(attrs, fq)
            self.freqs[attrs] = fq
        if vals in self.freqs[attrs]:
            return float(self.freqs[attrs][vals])
        return self.interpolators[attrs](vals) 

    def _build_interpolator(self, attrs, freqs):
        if len(attrs) == 1:
            sorted_keys = sorted(k[0] for k in freqs.keys())
            a = np.array(sorted_keys)
            v = np.array([freqs[(k,)] for k in sorted_keys])
            interp_fun = interp1d(a, v, copy=False, bounds_error=False, fill_value=0.0)
            self.interpolators[attrs] = lambda x: interp_fun(x)[0]
        else:
            a = np.array(freqs.keys())
            v = np.array(freqs.values())
            #border_points = list(self._calculate_border_points(a))
            #a = np.append(a, border_points, axis=0)
            #v = np.append(v, [[0.0]] * len(border_points))
            #self.interpolators[attrs] = LinearNDInterpolator(a, v, 0.0)
            self.interpolators[attrs] = NearestNDInterpolator(a, v)

    def _calculate_border_points(self, arr):
        points = (arr.min(axis=0), arr.max(axis=0))
        for indices in product((0,1), repeat=len(points[0])):
            yield tuple(points[i][j] for j,i in enumerate(indices))

    def _kirkwood_approx(self, attrs, vals):
        return reduce(lambda acc, val: (val / acc) if acc > 0.0 else 0.0,
                (reduce(operator.mul,
                    (self._density((attrs[i] for i in indices),
                        (vals[i] for i in indices))
                        for indices in combinations(range(len(attrs)), n)))
                    for n in xrange(1,len(attrs))))
