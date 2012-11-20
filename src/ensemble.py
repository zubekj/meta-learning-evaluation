import Orange

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

    def __call__(self, instance):
        return self.classifiers[self.rand(len(self.classifiers))](instance)

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

    def __call__(self, instance):
        def index_max(values):
            return max(xrange(len(values)), key=values.__getitem__)
        return self.classifiers[index_max(map(self.scoring_fun,
                                              self.classifiers))](instance)

class WeightedVoteClassifier(Orange.classification.Classifier):
    """
    Returns the class which received the most support from the classifiers
    passed to the constructor. Support from a single classifier is
    proportional to its score. Scoring function has to be provided.
    """

    def __init__(self, classifiers, scoring_fun,
                 name="BestDecidesClassifier"):
        self.name = name
        self.classifiers = classifiers
        self.scoring_fun = scoring_fun

    def __call__(self, instance):
        votes = {}
        for c in self.classifiers:
            cls = c(instance)
            ncls = cls.native()
            if ncls not in votes:
                votes[ncls] = [cls, 0]
            votes[ncls][1] += self.scoring_fun(c) 
        return max(votes.values(), key=lambda x: x[1])[0]
