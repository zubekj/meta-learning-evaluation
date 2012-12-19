import sys
import Orange
import cPickle

sys.path.append('../')

from utils.cSimilarity import *

if __name__ == '__main__':

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "iris"

    data = Orange.data.Table(data_file)

    rand = Orange.misc.Random(0)

    min_data = build_min_subsets_list_mc(data, range(len(data)), rand)
    max_data = build_max_subsets_list_mc(data, range(len(data)), rand)

    data_path = "{0}_minmax_subsets.pkl".format(data_file)

    data_file = open(data_path, "wb")
    cPickle.dump(min_data, data_file)
    cPickle.dump(max_data, data_file)
    data_file.close()
