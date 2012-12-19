import sys
import unittest
import copy
import Orange

sys.path.append('../../src/experiments_real_data/')
sys.path.append('../../src/')

from benchmark import *
from utils.cSimilarity import *
#from utils.similarity import *

class TestBenchmark(unittest.TestCase):

    def test_build_set_list_desc_similarity(self):
        data = Orange.data.Table("test.tab")

        data1 = [data[0], data[1]]
        data2 = [data[3], data[1]]
        data3 = [data[3], data[2]]
        
        l = build_set_list_desc_similarity(data, 0.5)
        self.assertEqual(list(l[0]), data1)
        self.assertEqual(list(l[1]), data2)
        self.assertEqual(list(l[2]), data3)

    def test_build_set_list_desc_similarity_long(self):
        data = Orange.data.Table("iris")
        def test_metric(metric_fun):
            l = build_set_list_desc_similarity(data, 0.5, metric_fun)
            dists = [datasets_distance(l[0], x, metric_fun) for x in l]
            for i in xrange(1,len(dists)):
                self.assertGreaterEqual(dists[i], dists[i-1])
            self.assertGreater(dists[-1], dists[0])
        test_metric(hamming)
        test_metric(euclidean)

if __name__ == '__main__':
    unittest.main()
