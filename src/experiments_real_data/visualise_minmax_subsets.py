import sys
from itertools import izip
import cPickle

if __name__ == '__main__':

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "iris"

    data_path = "{0}_minmax_subsets.pkl".format(data_file)

    data_file = open(data_path, "rb")
    min_sets, min_dists = cPickle.load(data_file)
    max_sets, max_dists = cPickle.load(data_file)
    data_file.close()

    print "Size Min_dist Max_dist"
    for s1, d1, s2, d2 in izip(min_sets, min_dists, max_sets, max_dists):
        i = sum(s1)
        if i != sum(s2):
            print "Inconsistent data"
            exit(1)
        print "{0} {1} {2}".format(i, d1, d2)
