import sys
import unittest

sys.path.append('../../src/experiments_real_data/')
sys.path.append('../../src/')

import sdistances

class TestSDistances(unittest.TestCase):

    def test_distance_conversion(self):
        converter = sdistances.HDistanceConverter('iris.tab')
        
        self.assertEqual(converter.subset_size(1.0), 1)
        self.assertEqual(converter.subset_size(0.000001), 150)
        self.assertEqual(converter.subset_size(0.0), 150)

if __name__ == '__main__':
    unittest.main()
