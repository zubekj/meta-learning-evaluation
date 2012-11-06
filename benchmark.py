import sys
import math
import cPickle
import Orange
import Orange.classification.neural
from Orange.evaluation.testing import learn_and_test_on_test_data

import datasets

def select_random_features(data, test_data, n, random_generator=Orange.misc.Random(0)):
    """
    Returns new data table with n random features selected from the given table.
    """
    features_number = len(data.domain) - 1
    if n >= features_number:
        return (data, test_data)
    indices = range(features_number)
    for i in range(features_number - n):
        del indices[random(len(indices))]
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
        idx = random(len(indices_1))
        indices_2.append(indices_1[idx])
        del indices_1[idx]
    t1 = data.get_items_ref(indices_1)
    t2 = data.get_items_ref(indices_2)
    return (t2, t1)

data_sets = datasets.get_datasets() 

learning_proportion = 0.7
#learn_subsets = [1.0, 0.3, 0.2, 0.1, 0.075, 0.05]
learn_subsets = [x * 0.1 for x in range(1, 10)]
sample_size = 10
#feature_subsets = [1.0, 0.8, 0.6, 0.4, 0.2]
feature_subsets = [1.0]

learners = [#nbdisc.Learner(name="bayes"),
            Orange.classification.bayes.NaiveLearner(name="bayes"),
            Orange.classification.knn.kNNLearner(name="knn"),
            Orange.classification.svm.MultiClassSVMLearner(name="svm"),
            Orange.classification.tree.SimpleTreeLearner(name="tree"),
            Orange.classification.neural.NeuralNetworkLearner(),
            Orange.classification.majority.MajorityLearner(name="majority")]

random = Orange.misc.Random(0)

results = {}

if len(sys.argv) > 1:
    data_file = sys.argv[1]
else:
    data_file = "iris"

# Levels: 1. Learn subset, 2. Feature subset, 3. Learning algorithm
data = Orange.data.Table(data_file)
learn_data, test_data = split_dataset_random(data, learning_proportion)
for sp in learn_subsets:
    results[sp] = {}
    for fs in feature_subsets:
        results[sp][fs] = {}
        for i in range(sample_size):
            sp_ldata, _n = split_dataset_random(data, sp)
            fs_ldata, fs_tdata = select_features_proportion(sp_ldata, test_data, fs, random)
            results[sp][fs][i] = learn_and_test_on_test_data(learners, fs_ldata, fs_tdata)

learners_names = map(lambda x: x.name, learners)

data_path = "{0}_data.pkl".format(data_file)

data_file = open(data_path, "wb")
cPickle.dump(learners_names, data_file)
cPickle.dump(results, data_file)
data_file.close()
