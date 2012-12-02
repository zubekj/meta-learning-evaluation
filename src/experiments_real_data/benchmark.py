import sys
import math
import random
import cPickle
import Orange
import neural
from itertools import imap
from operator import itemgetter
from Orange.evaluation.testing import learn_and_test_on_test_data
from Orange.classification.svm import kernels

sys.path.append('~/projects/meta-learning-evaluation/src/utils/')

from similarity import instance_dataset_distance, hamming

import datasets

class OrangeRandom(random.Random):
    
    def __init__(self, orngRand):
        self.rand = orngRand
    
    def randint(self, a, b):
        if b <= a:
            return 0
        return self.rand(b - a) + a

rand = Orange.misc.Random(0)

# Global options
LEARNING_PROPORTION = 0.7
LEARN_SUBSETS = [1 - math.log(x, 11) for x in xrange(10, 0, -1)] # Log scale
SAMPLE_SIZE = 10
FEATURE_SUBSETS = [1.0]
LEARNERS = [Orange.classification.bayes.NaiveLearner(name="bayes"),
            Orange.classification.knn.kNNLearner(name="knn"),
            Orange.classification.svm.SVMLearner(kernel_type=kernels.RBF,
                                                 name="svm_rbf"),
            Orange.classification.svm.SVMLearner(kernel_type=kernels.Linear,
                                                 name="svm_linear"),
            Orange.classification.svm.SVMLearner(kernel_type=kernels.Polynomial,
                                                 name="svm_polynomial"),
            Orange.classification.svm.SVMLearner(kernel_type=kernels.Sigmoid,
                                                 name="svm_sigmoid"),
            Orange.classification.tree.SimpleTreeLearner(name="tree"),
            neural.NeuralNetworkLearner(name="neural_net",
                                        rand=OrangeRandom(rand)),
            Orange.classification.majority.MajorityLearner(name="majority")
            ]

def select_random_features(data, test_data, n, random_generator=Orange.misc.Random(0)):
    """
    Returns new data table with n random features selected from the given table.
    """
    features_number = len(data.domain) - 1
    if n >= features_number:
        return (data, test_data)
    indices = range(features_number)
    for i in range(features_number - n):
        del indices[random_generator(len(indices))]
    sel = indices + [features_number]
    return (data.select(sel), test_data.select(sel))

def select_features_proportion(data, test_data, p,
        random_generator=Orange.misc.Random(0)):
    """
    Returns new data table with n random features selected, where
    n = len(data) * p.
    """
    return select_random_features(data, test_data,
            int(math.ceil(len(data.domain) * p)), random_generator)
         
def split_dataset(data, p):
    """
    Splits the data table according to given proportion.
    """
    l = len(data)
    t1 = data.get_items_ref(range(int(math.floor(p*l))))
    t2 = data.get_items_ref(range(int(math.ceil(p*l)), l))
    return (t1, t2)

def split_dataset_random(data, p, random_generator=Orange.misc.Random(0)):
    """
    Randomly selects instances from the data table and divides them into
    two tables according to given proportion.
    """
    l = len(data)
    indices_1 = range(l)
    indices_2 = []
    for i in range(int(math.floor(p*l))):
        idx = random_generator(len(indices_1))
        indices_2.append(indices_1[idx])
        del indices_1[idx]
    t1 = data.get_items_ref(indices_1)
    t2 = data.get_items_ref(indices_2)
    return (t2, t1)

def build_set_list_desc_similarity(data, set_size, metric=hamming,
                                   rand=Orange.misc.Random(0)):
    """
    Builds a list of subsets of data in which each consecutive subset is less
    similar to the first one (uses utils.similarity.datasets_distance). Each
    subset is of size S = set_size * len(data).
    """
    def distance_to_s0(x):
        return instance_dataset_distance(x, s0, metric)
    s0, _ = split_dataset_random(data, set_size, rand)
    asc_list = sorted(data, key=distance_to_s0)
    sets = [s0]
    for i in xrange(len(s0), len(asc_list)):
        s = sets[-1].get_items(range(len(sets[-1])))
        idx = min(enumerate(imap(distance_to_s0, s)), key=itemgetter(1))[0] 
        s[idx] = asc_list[i]
        sets.append(s)
    return sets

def benchmark_features_and_data_subsets(data, rand):
    # Levels: 1. Learn subset, 2. Feature subset (3. Samples, 4. Learner)
    levels = 2
    learn_data, test_data = split_dataset_random(data, LEARNING_PROPORTION, rand)
    for sp in LEARN_SUBSETS:
        results[sp] = {}
        for fs in FEATURE_SUBSETS:
            results[sp][fs] = {}
            for i in range(SAMPLE_SIZE):
                sp_ldata, _n = split_dataset_random(data, sp)
                fs_ldata, fs_tdata = select_features_proportion(sp_ldata, test_data, fs, rand)
                results[sp][fs][i] = learn_and_test_on_test_data(LEARNERS, fs_ldata, fs_tdata)
    return (levels, results)



if __name__ == '__main__':


    results = {}

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "iris"

    data = Orange.data.Table(data_file)

    levels, results = benchmark_features_and_data_subsets(data, rand)

    learners_names = map(lambda x: x.name, learners)

    data_path = "{0}_data.pkl".format(data_file)

    data_file = open(data_path, "wb")
    cPickle.dump(learners_names, data_file)
    cPickle.dump(levels, data_file)
    cPickle.dump(results, data_file)
    data_file.close()
