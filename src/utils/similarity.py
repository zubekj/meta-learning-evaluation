import operator
from itertools import imap, combinations
from math import sqrt, log

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

def data_distribution(data):
    """
    Calculates a discrete distribution of values in the dataset. All the
    possible combinations of attributes are taken into account.
    """
    n_attrs = len(data.domain)
    n_vals = len(data)
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
    for sdistr in distr.values():
        for val in sdistr:
            sdistr[val] = float(sdistr[val]) / n_vals
    return distr

def hellinger_distances_sum(cdistr1, cdistr2):
    """
    Sum of Hellinger distances for two sets of analogous distributions.
    """
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

def hellinger_distance(distr1, distr2):
    """
    Calculates Hellinger distance between two discrete probability
    distributions.
    """
    s = 0
    for v in distr1:
        if v in distr2:
            s += (distr1[v] - distr2[v]) * (distr1[v] - distr2[v])
        else:
            s += distr1[v] * distr1[v]
    for v in distr2:
        if v not in distr1:
            s += distr2[v] * distr2[v]
    return sqrt(s)/sqrt(2)

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
