import sys
import unittest
import copy
import Orange

sys.path.append('../../src/experiments_real_data/')
sys.path.append('../../src/utils/')

from benchmark import *
from similarity import datasets_distance, hamming

class TestBenchmark(unittest.TestCase):

    def test_build_set_list_desc_similarity(self):
        data = Orange.data.Table("test.tab")

        data1 = [data[1], data[2]]
        data2 = [data[3], data[2]]
        data3 = [data[3], data[0]]
        
        l = build_set_list_desc_similarity(data, 0.5)
        self.assertEqual(list(l[0]), data1)
        self.assertEqual(list(l[1]), data2)
        self.assertEqual(list(l[2]), data3)

    def test_build_set_list_desc_similarity_long(self):
        data = Orange.data.Table("iris")
        l = build_set_list_desc_similarity(data, 0.5)
        dists = [datasets_distance(l[0], x, hamming) for x in l]
        for i in xrange(1,len(dists)):
            self.assertGreaterEqual(dists[i], dists[i-1])
        self.assertGreater(dist[-1], dist[0])

if __name__ == '__main__':
    unittest.main()
