import sys
import unittest
import Orange

sys.path.append('../../src/')

from utils.distribution import *

class TestDistribution(unittest.TestCase):

    def test_density(self):
        data = Orange.data.Table('iris')
        jd = JointDistributions(data)
        self.assertEqual(jd.density({0: 4.3}), 1.0)
        self.assertEqual(jd.density({4: 0}), 50.0)
        self.assertEqual(jd.density({0: 5.0, 1: 2.0}), 1.0)
        self.assertEqual(jd.density({0: 6.0, 1: 2.0}), 0.0)
        self.assertEqual(jd.density({0: 6.0, 4: 1.0}), 4.0)

if __name__ == '__main__':
    unittest.main()
