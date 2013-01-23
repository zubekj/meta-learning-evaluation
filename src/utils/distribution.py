import operator
from math import sqrt
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
        Constructs an instance of JointDistributions from the given data set.

        Args:
            data: data set, instance of Orange.data.Table,
            kirkwood_level: minimum order of joint to apply Kirkwood approximation.
        """
        self.kirkwood_level = kirkwood_level
        self.data = data
        self.interpolators = {}
        self.freqs = {}

    def density(self, vals):
        """
        Returns approximated joint probability density function value for
        the given instance. Returned values are not normalized.
        
        Args:
            vals: values of attributes.
        Returns:
            density function value.
        """
        return self._density(xrange(len(vals)), (round(float(v), 4) for v in vals))

    def margin_density(self, attrs_vals):
        """
        Returns approximated marginal probability density function value for
        the given set of attributes values. Returned values are not normalized.
        
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
            r = self._kirkwood_approx(attrs, vals)
        return r

    def _freqs_density(self, attrs, vals):
        if not attrs in self.freqs:
            fq = {}
            for d in self.data:
                key = tuple(round(float(d[a]), 4) for a in attrs)
                if key not in fq:
                    fq[key] = 0
                fq[key] += 1
            #self._build_interpolator(attrs, fq)
            self.freqs[attrs] = fq
        if vals in self.freqs[attrs]:
            return float(self.freqs[attrs][vals])
        return 0.0
        #return self.interpolators[attrs](vals) 

    def _build_interpolator(self, attrs, freqs):
        if len(attrs) == 1:
            sorted_keys = sorted(k[0] for k in freqs.keys())
            a = np.array(sorted_keys)
            v = np.array([freqs[(k,)] for k in sorted_keys])
            if len(v) > 1:
                interp_fun = interp1d(a, v, copy=False, bounds_error=False, fill_value=0.0)
                self.interpolators[attrs] = lambda x: interp_fun(x)[0]
            else:
                self.interpolators[attrs] = lambda x: v[0]
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


def hellinger_distance(distr1, distr2, data):
    data = data.getItems(range(len(data)))
    data.remove_duplicates()
    e1 = np.array([distr1.density(d) for d in data])
    e1 = e1 / np.sum(e1)
    e2 = np.array([distr2.density(d) for d in data])
    e2 = e2 / np.sum(e2)
    r = np.sqrt(e1) - np.sqrt(e2)
    return np.sqrt(np.sum(np.multiply(r,r))/2)

####

#MC_ITERATIONS = 50
MC_ITERATIONS = 10

def indices_gen(p, rand, data):
    n = len(data)
    if p == n:
        return [0] * p
    if p == 1:
        ind = [1] * n
        ind[rand(n)] = 0
        return ind
    indices2 = Orange.data.sample.SubsetIndices2(p0=p)
    indices2.random_generator = rand
    return indices2(data)

def random_subset_dist(ddata, ddata_distr, n, kirkwood_level, rand=Orange.misc.Random(0)):
    """
    Draws a random subset of size n from the data and returns it along with its Hellinger
    distance from the whole dataset. Subset is represented with binary mask; 1 at i-th place
    means that i-th example belongs to the subset.
    """
    n = int(n)
    smask = indices_gen(n, rand, ddata)
    sdata = ddata.select(smask, 0)
    sdata_distr = JointDistributions(sdata, kirkwood_level)
    return smask, hellinger_distance(sdata_distr, ddata_distr, ddata)

def build_minmax_subsets_list_mc(data, subset_sizes = None, rand=Orange.misc.Random(0)):
    """
    Builds a list of subsets of different sizes minimizing and maximazing Hellinger
    distance from the whole dataset. Uses Monte Carlo approach.
    """
    if not subset_sizes:
        subset_sizes = range(len(data)+1)
    
    ddata = Orange.data.discretization.DiscretizeTable(data,
                   method=Orange.feature.discretization.EqualWidth(n=len(data)/10))
    level = 5
    ddata_distr = JointDistributions(ddata, kirkwood_level=level)

    min_subsets_list = []
    max_subsets_list = []

    for i in subset_sizes:
        min_subset, min_d = random_subset_dist(ddata, ddata_distr,
                                               i, level, rand)
        max_subset, max_d = min_subset, min_d
        for j in range(MC_ITERATIONS-1):
            subset, d = random_subset_dist(ddata, ddata_distr,
                                           i, rand)
            if d < min_d:
                min_subset = subset
                min_d = d
            if d > max_d:
                max_subset = subset
                max_d = d
        min_subsets_list.append((min_subset, min_d))
        max_subsets_list.append((max_subset, max_d))
    return (min_subsets_list, max_subsets_list)
