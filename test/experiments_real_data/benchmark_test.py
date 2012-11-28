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
        data1 = data.get_items_ref([0,1])
        data2 = data.get_items_ref([2,1])
        data3 = data.get_items_ref([2,3])
        l = build_set_list_desc_similarity(data, 0.5)
        self.assertEqual(l[0][0], data1[0])

if __name__ == '__main__':
    unittest.main()
