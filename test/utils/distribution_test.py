import sys
import unittest
import Orange
import numpy as np

sys.path.append('../../src/')

from utils.distribution import *

class TestDistribution(unittest.TestCase):

    def test_density(self):
        data = Orange.data.Table('iris')
        jd = JointDistributions(data)
        self.assertEqual(jd.margin_density({0: 7.9}), 1.0)
        self.assertEqual(jd.margin_density({0: 4.3}), 1.0)
        self.assertEqual(jd.margin_density({4: 0}), 50.0)
        self.assertEqual(jd.margin_density({0: 5.0, 1: 2.0}), 1.0)
        self.assertEqual(jd.margin_density({0: 6.0, 1: 2.0}), 2.0)
        self.assertEqual(jd.margin_density({0: 6.0, 4: 1.0}), 4.0)
        self.assertAlmostEqual(jd.margin_density({0: 6.0, 1: 4.0, 2: 3.0}), 0.16666666)
        self.assertAlmostEqual(jd.margin_density({0: 6.0, 1: 2.0, 2: 3.0}), 0.33333333)
        self.assertAlmostEqual(jd.margin_density({0: 6.0, 1: 0.0, 2: 3.0}), 0.0)
    
    def test__calculate_border_points(self):
        data = Orange.data.Table('iris')
        jd = JointDistributions(data)
        a = np.array([(2,1,0), (-2,2,1), (1,-2,2), (0,0,-2)])
        l = list(jd._calculate_border_points(a))
        self.assertEqual(l, [(-2,-2,-2), (-2,-2,2), (-2,2,-2), (-2,2,2), (2,-2,-2),
                             (2,-2,2), (2,2,-2), (2,2,2)])

if __name__ == '__main__':
    unittest.main()
