import sys
import math
import Orange
import cPickle

sys.path.append('../')

from utils.cSimilarity import *

LEARN_SUBSETS = [1 - math.log(x, 11) for x in xrange(10, 0, -1)] # Log scale

def generate_sizes(dlen):
    return range(1, int(LEARN_SUBSETS[0] * dlen)) + [p * dlen for p in LEARN_SUBSETS]

if __name__ == '__main__':

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "iris"

    data = Orange.data.Table(data_file)

    rand = Orange.misc.Random(0)

    subset_sizes = generate_sizes(len(data))
    min_data = build_min_subsets_list_mc(data, subset_sizes, rand)
    max_data = build_max_subsets_list_mc(data, subset_sizes, rand)

    data_path = "{0}_minmax_subsets.pkl".format(data_file)

    data_file = open(data_path, "wb")
    cPickle.dump(min_data, data_file)
    cPickle.dump(max_data, data_file)
    data_file.close()
