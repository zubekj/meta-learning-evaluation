import sys
import unittest
import copy
import Orange
from Orange.classification import ConstantClassifier

sys.path.append('../../src/')

from utils.cSimilarity import *

class TestSimilarity(unittest.TestCase):

    def test_hamming(self):
        a = [1,0,2]
        b = [1,1,2]
        self.assertEqual(hamming(a,a), 0)
        self.assertEqual(hamming(b,b), 0)
        self.assertEqual(hamming(a,b), 1)

    def test_hamming_iris(self):
        data = Orange.data.Table('iris')
        a = data[0]
        b = data[1]
        self.assertEqual(hamming(a,a), 0)
        self.assertEqual(hamming(b,b), 0)
        self.assertEqual(hamming(a,b), 2)

    def test_euclidean(self):
        a = [1,0,0]
        b = [1,4,3]
        self.assertEqual(euclidean(a,a), 0)
        self.assertEqual(euclidean(b,b), 0)
        self.assertEqual(euclidean(a,b), 5)

    def test_euclidean_iris(self):
        data = Orange.data.Table('iris')
        a = data[0]
        b = data[1]
        self.assertEqual(euclidean(a,a), 0)
        self.assertEqual(euclidean(b,b), 0)
        self.assertAlmostEqual(euclidean(a,b), 0.5385164)

    def test_datasets_distance(self):
        data1 = Orange.data.Table("test.tab")
        data2 = Orange.data.Table("test1.tab")
        self.assertEqual(datasets_distance(data1, data1, hamming), 0)
        self.assertEqual(datasets_distance(data1, data2, hamming), 2)

    def test_instance_dataset_distance(self):
        data1 = Orange.data.Table("test.tab")
        i1 = data1[0]
        data2 = Orange.data.Table("test1.tab")
        i2 = data2[3]
        self.assertEqual(instance_dataset_distance(i1, data1, hamming), 0)
        self.assertEqual(instance_dataset_distance(i2, data1, hamming), 1)

    def test_data_distribution(self):
        data1 = Orange.data.Table("test.tab")
        distr = data_distribution(data1)
        self.assertAlmostEqual(distr[(0,)][(1,)], 0.071428571)
        self.assertAlmostEqual(distr[(2,)][('1',)], 0.035714285)

    def test_kl_divergence(self):
        data1 = Orange.data.Table("test.tab")
        data2 = Orange.data.Table("test1.tab")
        cdistr1 = data_distribution(data1)
        cdistr2 = data_distribution(data2)
        self.assertAlmostEqual(kl_divergence(cdistr1[(0,)], cdistr2[(0,)]), 0.0296455356)

    def test_hellinger_distance(self):
        data1 = Orange.data.Table("test.tab")
        data2 = Orange.data.Table("test1.tab")
        distr1 = data_distribution(data1)
        distr2 = data_distribution(data2)
        empty_data_distr = data_distribution(Orange.data.Table(data1.domain))
        self.assertEqual(hellinger_distance(distr1, distr1), 0)
        self.assertAlmostEqual(hellinger_distance(distr1, empty_data_distr), 0.70710678)
        self.assertAlmostEqual(hellinger_distance(distr1, distr2), 0.2804031267)
        
    def test_hellinger_distance_subset(self):
        data1 = Orange.data.Table("test.tab")
        data2 = data1.get_items(range(len(data1)))
        data2.append(data2[0])
        distr1 = data_distribution(data1)
        distr2 = data_distribution(data2)
        distr2_sum = cdistr_total_sum(distr2)
        self.assertAlmostEqual(hellinger_distance(distr1, distr2),
                    hellinger_distance_subset(distr1, distr2, distr2_sum))

    def test_build_set_list_dec_dist(self):
        data = Orange.data.Table("test.tab")
        l, d = build_subsets_dec_dist(data)
        for i in xrange(1,len(l)):
            self.assertGreater(sum(l[i]), sum(l[i-1]))

##    def test_build_set_list_dec_dist_long(self):
##        data = Orange.data.Table("iris")
##        l, d = build_subsets_dec_dist(data)
##        for i in xrange(1,len(l)):
##            self.assertGreater(sum(l[i]), sum(l[i-1]))
#
    def test_build_max_subsets_list_mc(self):
        data = Orange.data.Table("test.tab")
        l = build_max_subsets_list_mc(data)
        for i in xrange(1,len(l)):
            self.assertGreater(sum(l[i-1][0]), sum(l[i][0]))

    def test_build_min_subsets_list_mc(self):
        data = Orange.data.Table("test.tab")
        l = build_min_subsets_list_mc(data)
        for i in xrange(1,len(l)):
            self.assertGreater(sum(l[i-1][0]), sum(l[i][0]))

#    def test_build_min_subsets_list_mc_long(self):
#        data = Orange.data.Table("iris")
#        l = build_min_subsets_list_mc(data)
#        for i in xrange(1,len(l)):
#            self.assertGreater(sum(l[i-1][0]), sum(l[i][0]))

if __name__ == '__main__':
    unittest.main()
