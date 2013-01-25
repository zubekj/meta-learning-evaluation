import operator
import random
from collections import defaultdict
from math import sqrt, factorial
from itertools import combinations, product
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d, NearestNDInterpolator
import Orange

class JointDistributions():
    """
    Class for representing and approximating joint distributions of attributes
    over a data set.
    """

    def __init__(self, data, kirkwood_level=None):
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
        self.k_cache = {}

    def density(self, vals):
        """
        Returns approximated joint probability density function value for
        the given instance. Returned values are not normalized.
        
        Args:
            vals: values of attributes.
        Returns:
            density function value.
        """
        return self._density(xrange(len(vals)), (float(v) for v in vals))

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
        if self.kirkwood_level and len(attrs) < self.kirkwood_level:
            r = self._freqs_density(attrs, vals)
        else:
            r = self._kirkwood_approx(attrs, vals)
        return r

    def _freqs_density(self, attrs, vals):
        if not attrs in self.freqs:
            fq = defaultdict(int)
            for d in self.data[:,list(attrs)]:
                fq[tuple(d)] += 1
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
        k = (attrs, vals)
        if not k in self.k_cache:
            r = reduce(lambda acc, val: (val / acc) if acc > 0.0 else 0.0,
                    (reduce(operator.mul,
                        (self._density((attrs[i] for i in indices),
                            (vals[i] for i in indices))
                            for indices in combinations(range(len(attrs)), n)))
                        for n in xrange(1,len(attrs))))
            self.k_cache[k] = r
            return r
        return self.k_cache[k]


def hellinger_distance(distr1, distr2, data):
    e1 = np.array([distr1.density(d) for d in data])
    e1 = e1 / np.sum(e1)
    e2 = np.array([distr2.density(d) for d in data])
    e2 = e2 / np.sum(e2)
    r = np.sqrt(e1) - np.sqrt(e2)
    return np.sqrt(np.sum(np.multiply(r,r))/2)

####

MC_ITERATIONS = 50
#MC_ITERATIONS = 10

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

def combined_distribution(distr, level, distr_space):
    indices = range(len(distr.data[0]))
    l = []
    for j in xrange(1,level+1):
        for c in combinations(indices, j):
            for d in distr_space[:,list(c)]:
                l.append(distr._freqs_density(c, tuple(d)))
    return np.array(l)

def random_subset_dist(ddata, distr_space, dd_sq_vals, n, level):
    """
    Draws a random subset of size n from the data and returns it along with its Hellinger
    distance from the whole dataset. Subset is represented with binary mask; 2 at i-th place
    means that i-th example belongs to the subset.
    """
    n = int(n)
    sdata = ddata[random.sample(xrange(len(ddata)), n)]
    sdata_distr = JointDistributions(sdata)
    sd_vals = combined_distribution(sdata_distr, level, distr_space)
    sd_vals /= np.sum(sd_vals)
    r = np.sqrt(sd_vals) - dd_sq_vals
    dist = np.sqrt(np.sum(np.multiply(r,r))/2)
    return dist

def build_minmax_subsets_list_mc(data, level, subset_sizes = None):
    """
    Builds a list of subsets of different sizes minimizing and maximazing Hellinger
    distance from the whole dataset. Uses Monte Carlo approach.
    """
    if not subset_sizes:
        subset_sizes = range(len(data)+1)
   
    if level > len(data.domain):
        level = len(data.domain)

    ddata = Orange.data.discretization.DiscretizeTable(data,
                   method=Orange.feature.discretization.EqualWidth(n=len(data)/10))
    ddata = np.array([tuple(float(d[i]) for i in xrange(len(ddata.domain))) for d in ddata])
    ddata_distr = JointDistributions(ddata)

    def unique_rows(a):
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

    distr_space = unique_rows(ddata)
    
    dd_sq_vals = combined_distribution(ddata_distr, level, distr_space)
    dd_sq_vals /= np.sum(dd_sq_vals)
    dd_sq_vals = np.sqrt(dd_sq_vals)
    
    min_subsets_list = []
    max_subsets_list = []

    for i in subset_sizes:
        min_d = random_subset_dist(ddata, distr_space,
                                               dd_sq_vals,
                                               i, level)
        max_d = min_d
        for j in range(MC_ITERATIONS-1):
            d = random_subset_dist(ddata, distr_space,
                                           dd_sq_vals,
                                           i, level)
            if d < min_d:
                min_d = d
            if d > max_d:
                max_d = d
        min_subsets_list.append(min_d)
        max_subsets_list.append(max_d)
    return (min_subsets_list, max_subsets_list)
