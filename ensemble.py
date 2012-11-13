import Orange

class MajorityVoteClassifier(Orange.ensemble.bagging.BaggedClassifier):
    """
    MajorityVoteClassifier returns class which received the most votes
    from classifiers passed to the constructor. This class is a wrapper
    for Orange.ensemble.bagging.BaggedClassifier.
    """

    def __init__(self, classifiers, name="MajorityVoteClassifier",
                 class_var=None):
        if not class_var:
            class_var = classifiers[0].class_var
        super(self, MajorityVoteClassifier).__init__(classifiers, name,
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
        classifiers[self.rand(len(classifiers))](instance)
