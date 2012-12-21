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
    min_sets = cPickle.load(data_file)
    max_sets = cPickle.load(data_file)
    data_file.close()

    print "Size Min_dist Max_dist"
    for min_s, max_s in izip(min_sets, max_sets):
        i = len(min_s[0]) - sum(min_s[0])
        if i != len(max_s[0]) - sum(max_s[0]):
            print "Inconsistent data"
            exit(1)
        print "{0} {1} {2}".format(i, min_s[1], max_s[1])
