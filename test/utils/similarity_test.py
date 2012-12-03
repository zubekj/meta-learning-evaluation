import sys
import unittest
import copy
import Orange
from Orange.classification import ConstantClassifier

sys.path.append('../../src/')

from utils.similarity import *

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

if __name__ == '__main__':
    unittest.main()
