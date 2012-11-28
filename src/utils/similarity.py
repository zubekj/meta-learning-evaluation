import operator
from itertools import imap
from math import sqrt

def datasets_distance(set1, set2, metric_function):
    """
    Measures similarity between two datasets as D1 + D2, where:
    D1 -- sum of distances between each instance from set1 and the nearest
          instance from set2.
    D2 -- sum of distances between each instance from set2 and the nearest
          instance from set1.
    """
    def d(s1,s2):
        return sum(imap(lambda i1: min(imap(lambda i2: metric_function(i1, i2),
                                      s2)), s1))
    return d(set1, set2) + d(set2, set1)

def hamming(i1, i2):
    return sum(imap(operator.ne, i1, i2))

def euclidean(i1, i2):
    return sqrt(sum(imap(lambda x, y: (x-y)*(x-y), i1, i2)))
