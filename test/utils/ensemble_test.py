import sys
import unittest
import Orange
from Orange.classification import ConstantClassifier

sys.path.append('../../src/')

from utils.ensemble import *

class TestEnsemble(unittest.TestCase):

    def setUp(self):
        self.data = Orange.data.Table("test.tab")
        self.c1 = ConstantClassifier(self.data[0][2]) 
        self.c0 = ConstantClassifier(self.data[1][2]) 
        self.class1 = self.data[0][2]
        self.class0 = self.data[1][2]

    def test_random_decides(self):
        c = RandomDecidesClassifier([self.c0, self.c1],
                                    rand=Orange.misc.Random(1))
        i = self.data[0]
        self.assertEqual(c(i), self.class0)
        self.assertEqual(c(i), self.class1)
        self.assertEqual(c(i), self.class0)

    def test_majority_vote(self):
        c = MajorityVoteClassifier([self.c0, self.c1, self.c1])
        i = self.data[0]
        self.assertEqual(c(i), self.class1)
        c = MajorityVoteClassifier([self.c0, self.c0, self.c1])
        self.assertEqual(c(i), self.class0)

    def test_best_decides(self):
        def get_score(c):
            if c == self.c0:
                return 1
            else:
                return 0
        c = BestDecidesClassifier([self.c0, self.c1], get_score)
        i = self.data[0]
        self.assertEqual(c(i), self.class0)
 
    def test_weighted_vote(self):
        def get_score(c):
            if c == self.c0:
                return 0.4
            else:
                return 0.6
        i = self.data[0]
        c = WeightedVoteClassifier([self.c0, self.c1], get_score)
        self.assertEqual(c(i), self.class1)
        probs = c(i, Orange.classification.Classifier.GetProbabilities).values()
        self.assertAlmostEqual(probs[0], 0.4)
        self.assertAlmostEqual(probs[1], 0.6)
        c = WeightedVoteClassifier([self.c0, self.c1, self.c0], get_score)
        self.assertEqual(c(i), self.class0)

    def test_weighted_confidence(self):
        i = self.data[0]
        c = WeightedConfidenceSharingClassifier([self.c0, self.c1, self.c0])
        self.assertEqual(c(i), self.class0)
        probs = c(i, Orange.classification.Classifier.GetProbabilities).values()
        self.assertAlmostEqual(probs[0], 0.66666668)

    def test_threshold_classifier(self):
        c = MajorityVoteClassifier([self.c0, self.c1, self.c1, self.c1])
        i = self.data[0]
        probs = c(i, Orange.classification.Classifier.GetProbabilities).values()
        self.assertEqual(probs[0], 0.25)
        self.assertEqual(probs[1], 0.75)
        tc = ThresholdClassifier(c)
        self.assertEqual(c(i), self.class1)
        self.assertEqual(tc(i), self.class1)
        tc.threshold = 0.8
        self.assertEqual(tc(i), self.class0)


if __name__ == '__main__':
    unittest.main()
