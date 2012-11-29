import sys
import unittest
import copy
import Orange

sys.path.append('../../src/experiments_real_data/')
sys.path.append('../../src/utils/')

from benchmark import *

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

if __name__ == '__main__':
    unittest.main()
