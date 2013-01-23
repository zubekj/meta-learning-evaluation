import sys
import math
from matplotlib import pyplot
import Orange
import cPickle

sys.path.append('../')

from utils.distribution import *
#from utils.cSimilarity import *

#LEARN_SUBSETS = [1 - math.log(x, 11) for x in xrange(10, 0, -1)] # Log scale
LEARN_SUBSETS = [0.05 + 0.01*i for i in xrange(5)] + \
                [0.1+0.2*i for i in xrange(4)] + \
                [0.9 + 0.02*i for i in xrange(6)]
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
    min_data, max_data = build_minmax_subsets_list_mc(data, subset_sizes, rand)

    x = [int(size) for size in subset_sizes]
    y1 = [m[1] for m in min_data]
    y2 = [m[1] for m in max_data]

    pyplot.plot(x, y1, color="green") 
    pyplot.plot(x, y2, color="blue") 
    pyplot.show()
