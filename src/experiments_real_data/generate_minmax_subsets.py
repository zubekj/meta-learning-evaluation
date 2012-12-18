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

    min_data = build_subsets_dec_dist(data, True)
    max_data = build_subsets_dec_dist(data, False)

    data_path = "{0}_minmax_subsets.pkl".format(data_file)

    data_file = open(data_path, "wb")
    cPickle.dump(min_data, data_file)
    cPickle.dump(max_data, data_file)
    data_file.close()
