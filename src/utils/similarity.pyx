import operator
from itertools import imap, combinations
from math import log
from libc.math cimport sqrt
import Orange

def datasets_distance(set1, set2, metric_function):
    """
    Measures similarity between two datasets as D1 + D2, where:
    D1 -- sum of distances between each instance from set1 and the nearest
          instance from set2.
    D2 -- sum of distances between each instance from set2 and the nearest
          instance from set1.
    """
    def d(s1,s2):
        return sum(imap(lambda i1: instance_dataset_distance(i1, s2,
                                                    metric_function), s1))
    return d(set1, set2) + d(set2, set1)

def instance_dataset_distance(instance, dataset, metric_function):
    """
    Calculates the distance between instance and dataset as the
    distance from the given instance to the closest instance from the
    dataset.
    """
    return min(imap(lambda i2: metric_function(instance, i2), dataset))

def hamming(i1, i2):
    return sum(imap(operator.ne, i1, i2))

def euclidean(i1, i2):
    def diff(x, y):
        try:
            return x - y
        except TypeError:
            return int(x != y)

    return sqrt(sum(imap(lambda x, y: diff(x,y)**2, i1, i2)))

def data_distribution_nn(data):
    n_attrs = len(data.domain)
    indices = range(n_attrs)
    distr = {}
    for subset_size in xrange(1, n_attrs+1):
        for subset in combinations(indices, subset_size):
            sdistr = {}
            for d in data:
                val = tuple([d[i].value for i in subset])
                if val not in sdistr:
                    sdistr[val] = 0
                sdistr[val] += 1
            distr[subset] = sdistr
    return distr

def normalize_distribution(distr, n):
    return {subset: {val: (float(count) / n)
                          for (val, count) in sdistr.iteritems()}
                    for (subset, sdistr) in distr.iteritems()}

def data_distribution(data):
    """
    Calculates a discrete distribution of values in the dataset. All the
    possible combinations of attributes are taken into account.
    """
    n = len(data)
    return normalize_distribution(data_distribution_nn(data), n)
   
cpdef dict distribution_nn_add_instance(dict distr, instance):
    """
    Updates data distribution after adding a new instance.
    """
    for subset in distr:
        sdistr = distr[subset]
        val = tuple([instance[i].value for i in subset])
        if val not in sdistr:
            sdistr[val] = 0
        sdistr[val] += 1
    return distr

cpdef dict distribution_nn_remove_instance(dict distr, instance):
    """
    Updates data distribution after removing an instance.
    """
    for subset in distr:
        sdistr = distr[subset]
        val = tuple([instance[i].value for i in subset])
        sdistr[val] -= 1
    return distr

cpdef double hellinger_distances_sum(dict cdistr1, dict cdistr2):
    """
    Sum of Hellinger distances for two sets of analogous distributions.
    """
    cdef double s
    s = 0
    for k in cdistr1:
        if k in cdistr2:
            s += hellinger_distance(cdistr1[k], cdistr2[k])
        else:
            s += hellinger_distance(cdistr1[k], {})
    for k in cdistr2:
        if not k in cdistr1:
            s += hellinger_distance({}, cdistr2[k])
    return s

cpdef double hellinger_distance(dict distr1, dict distr2):
    """
    Calculates Hellinger distance between two discrete probability
    distributions.
    """
    cdef double s, a, b
    s = 0
    for v in distr1:
        a = distr1[v]
        if v in distr2:
            b = distr2[v]
            s += (a - b) * (a - b)
        else:
            s += a * a
    for v in distr2:
        if v not in distr1:
            b = distr2[v]
            s += b * b
    return sqrt(s/2.)

def kl_divergence(distr1, distr2):
    """
    Calculates Kullback-Leibner divergence between two discrete probability
    distributions. Divergence is well defined if every value from
    distr1 is included in distr2.
    
    distr1 can be treated as the "real" distribution and distr2 is
    is estimated distribution com
    """
    s = 0
    for v in distr1:
        if v in distr2:
            s += log(distr1[v] / distr2[v], 2) * distr1[v]
    return s
    
    
def build_subsets_dec_dist(data, minimalize=True):
    """
    Builds a list of subsets of the whole dataset iteratively using greedy approach
    based on Hellinger distance minimalization/maximalization. Each subset is represented
    as a list of 0's and 1's, 1 at i-th place means that i-th example belongs to the subset.
    
    TODO: Refactor
    """
    ddata = Orange.data.discretization.DiscretizeTable(data,
                   method=Orange.feature.discretization.EqualFreq(n=len(data)))

    data_distr = data_distribution(ddata)
    unassigned_data_ind = range(len(data))
    sets = [[0] * len(data)]
    sets_dists = []
    cdistr = data_distribution_nn(ddata.select(sets[-1], 1)) 
    while len(unassigned_data_ind):
        cset = ddata.select(sets[-1], 1)
        dists = []
        for i in unassigned_data_ind:
            cset.append(ddata[i])
            distribution_nn_add_instance(cdistr, ddata[i])
            dists.append(hellinger_distances_sum(normalize_distribution(cdistr,
                len(cset)), data_distr))
            del cset[-1]
            distribution_nn_remove_instance(cdistr, ddata[i])
        if minimalize:
            idx = min(xrange(len(dists)), key=dists.__getitem__)
        else:
            idx = max(xrange(len(dists)), key=dists.__getitem__)
        sets_dists.append(dists[idx])
        nset = sets[-1][:]
        nset[unassigned_data_ind[idx]] = 1
        sets.append(nset)
        distribution_nn_add_instance(cdistr, ddata[unassigned_data_ind[idx]])
        del unassigned_data_ind[idx]

    del sets[0]
    return sets, sets_dists
