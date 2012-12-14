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
            for d in data:
                val = tuple([subset] + [d[i].value for i in subset])
                if val not in distr:
                    distr[val] = 0
                distr[val] += 1
    for val in distr:
        distr[val] = float(distr[val]) / n_vals
    return distr

def kl_divergence(distr1=None, distr2=None):
    """
    Calculates Kullback-Leibner divergence between two discrete probability
    distributions. Divergence is well defined if every value from dist1 is
    included in distr2.
    """
    s = 0
    for v in distr1:
        if v in distr2:
            s += log(distr1[v] / distr2[v], 2) * distr1[v]
    return s
