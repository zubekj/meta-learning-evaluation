import sys
import math
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import Orange

sys.path.append('../')

from utils.distribution import *
#from utils.cSimilarity import *

#LEARN_SUBSETS = [1 - math.log(x, 11) for x in xrange(10, 0, -1)] # Log scale
LEARN_SUBSETS = [0.05 + 0.01*i for i in xrange(5)] + \
                [0.1+0.2*i for i in xrange(4)] + \
                [0.9 + 0.02*i for i in xrange(6)]
LEVEL = 2

def generate_sizes(dlen):
    return range(1, int(LEARN_SUBSETS[0] * dlen)) + [p * dlen for p in LEARN_SUBSETS]

def plot_levels(levels, subset_sizes, data_file):
    x = [int(size) for size in subset_sizes]
    colours = ["red", "green", "blue", "orange", "yellow", "black"]
    for l in levels:
        min_data, max_data = build_minmax_subsets_list_mc(data, l, subset_sizes)
        pyplot.plot(x, max_data, color=colours[l % len(colours)])
    pp = PdfPages('{0}_levels.pdf'.format(data_file))
    pp.savefig()
    pp.close()

def plot_minmax(level, subset_sizes, data_file):
    x = [int(size) for size in subset_sizes]
    min_data, max_data = build_minmax_subsets_list_mc(data, level, subset_sizes)
    pyplot.plot(x, min_data, color="green")
    pyplot.plot(x, max_data, color="red")
    pp = PdfPages('{0}_minmax.pdf'.format(data_file))
    pp.savefig()
    pp.close()
    print "Size Min_dist Max_dist"
    for i in xrange(len(x)):
        print x[i], min_data[i], max_data[i]

if __name__ == '__main__':

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "iris"

    random.seed(42)
    
    data = Orange.data.Table(data_file)
    subset_sizes = generate_sizes(len(data))

    #plot_levels(range(1,6), subset_sizes, data_file)
    plot_minmax(6, subset_sizes, data_file)
