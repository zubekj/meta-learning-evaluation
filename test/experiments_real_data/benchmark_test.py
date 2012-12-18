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

    def test_build_set_list_dec_dist(self):
        data = Orange.data.Table("test.tab")
        l, d = build_subsets_dec_dist(data)
        print d
        for i in xrange(1,len(l)):
            self.assertGreater(sum(l[i]), sum(l[i-1]))

    def test_build_set_list_dec_dist_long(self):
        data = Orange.data.Table("iris")
        l, d = build_subsets_dec_dist(data)
        for i in xrange(1,len(l)):
            self.assertGreater(sum(l[i]), sum(l[i-1]))

if __name__ == '__main__':
    unittest.main()
