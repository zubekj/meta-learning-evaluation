import Orange
from Orange.classification import Classifier

class MajorityVoteClassifier(Orange.ensemble.bagging.BaggedClassifier):
    """
    MajorityVoteClassifier returns the class which received the most votes
    from classifiers passed to the constructor. This class is a wrapper
    for Orange.ensemble.bagging.BaggedClassifier.
    """

    def __init__(self, classifiers, name="MajorityVoteClassifier",
                 class_var=None):
        if not class_var:
            class_var = classifiers[0].class_var
        super(MajorityVoteClassifier, self).__init__(classifiers, name,
                                                     class_var)

class RandomDecidesClassifier(Orange.classification.Classifier):
    """
    Returns the response of a random classifier from classifiers passed
    to the constructor.
    """

    def __init__(self, classifiers, rand=Orange.misc.Random(42),
                 name="RandomDecidesClassifier"):
        self.name = name
        self.classifiers = classifiers
        self.rand = rand

    def __call__(self, instance, result_type=Classifier.GetValue):
        return self.classifiers[self.rand(len(self.classifiers))](instance, result_type)

class BestDecidesClassifier(Orange.classification.Classifier):
    """
    Returns the response of the best classifier from classifiers passed
    to the constructor. Scoring function has to be provided.
    """

    def __init__(self, classifiers, scoring_fun,
                 name="BestDecidesClassifier"):
        self.name = name
        self.classifiers = classifiers
        self.scoring_fun = scoring_fun

    def __call__(self, instance, result_type=Classifier.GetValue):
        def index_max(values):
            return max(xrange(len(values)), key=values.__getitem__)
        return self.classifiers[index_max(map(self.scoring_fun,
                                              self.classifiers))](instance, result_type)

class WeightedVoteClassifier(Orange.classification.Classifier):
    """
    Returns the class which received the most support from the classifiers
    passed to the constructor. Support from a single classifier is
    proportional to its score. Scoring function has to be provided.
    """

    def __init__(self, classifiers, scoring_fun,
                 name="WeightedVoteClassifier"):
        self.name = name
        self.classifiers = classifiers
        self.scoring_fun = scoring_fun

    def __call__(self, instance, result_type=Classifier.GetValue):
        if instance.domain.class_vars:
            raise NotImplementedError
        
        cvals = instance.domain.class_var.values
        votes = [0] * len(cvals)
        for c in self.classifiers:
            cls = c(instance)
            votes[cvals.index(cls.native())] += self.scoring_fun(c)
        cprob = Orange.statistics.distribution.Discrete(votes)
        cprob.normalize()
        cval = Orange.data.Value(instance.domain.class_var, cprob.values().index(max(cprob)))

        if result_type == Classifier.GetValue:
            return cval
        elif result_type == Classifier.GetProbabilities:
            return cprob
        else:
            return [cval, cprob]

class WeightedConfidenceSharingClassifier(Orange.classification.Classifier):
    """
    Returns the class which received the most support from the classifiers
    passed to the constructor. Support from a single classifier is
    proportional to its confidence. Standard Orange probabilities estimations
    are used. Additional scoring function may be provided to scale probabilities
    differently for each classifier.
    """

    def __init__(self, classifiers, scoring_fun=None,
                 name="WeightedConfidenceSharingClassifier"):
        self.name = name
        self.classifiers = classifiers
        self.scoring_fun = scoring_fun
        if not self.scoring_fun:
            self.scoring_fun = lambda c, p: p

    def __call__(self, instance, result_type=Classifier.GetValue):
        if instance.domain.class_vars:
            raise NotImplementedError
        
        cvals = instance.domain.class_var.values
        votes = [0] * len(cvals)
        for c in self.classifiers:
            cls, prob = c(instance, Orange.classification.Classifier.GetBoth)
            mprob = max(prob)
            votes[cvals.index(cls.native())] += self.scoring_fun(c, mprob)
        cprob = Orange.statistics.distribution.Discrete(votes)
        cprob.normalize()
        cval = Orange.data.Value(instance.domain.class_var, cprob.values().index(max(cprob)))

        if result_type == Classifier.GetValue:
            return cval
        elif result_type == Classifier.GetProbabilities:
            return cprob
        else:
            return [cval, cprob]

class ThresholdClassifier(Orange.classification.Classifier):
    """
    Wrapper for constructed classifiers which allows to adjust threshold for binary
    classification. If the probability for positive class returned by base classifier
    is above the threshold positive is returned, otherwise negative.
    """

    def __init__(self, base_classifier, threshold=0.5, name="ThresholdClassifier"):
        self.name = name
        self.base_classifier = base_classifier
        self.threshold = threshold

    def __call__(self, instance, result_type=Classifier.GetValue):
        # Multi-class problems are not supported.
        domain = instance.domain
        if result_type == Classifier.GetProbabilities or domain.class_vars \
                or len(domain.class_var.values) > 2:
            return self.base_classifier(instance, result_type)
        
        cprob = self.base_classifier(instance, Classifier.GetProbabilities)
        cval = Orange.data.Value(domain.class_var, 1 if cprob[1] > self.threshold else 0)
        if result_type == Classifier.GetValue:
            return cval
        else:
            return [cval, cprob]
