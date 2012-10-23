import math
import Orange

import datasets

def split_dataset(data, p):
    l = len(data)
    t1 = data.get_items_ref(range(int(math.floor(p*l))))
    t2 = data.get_items_ref(range(int(math.ceil(p*l)), l))
    return (t1, t2)

def split_dataset_random(data, p, random_generator=Orange.misc.Random(0)):
    l = len(data)
    indices_1 = range(l)
    indices_2 = []
    for i in range(int(math.floor(p*l))):
        idx = random(len(indices_1))
        indices_2.append(indices_1[idx])
        del indices_1[idx]
    t1 = data.get_items_ref(indices_1)
    t2 = data.get_items_ref(indices_2)
    return (t1, t2)

data_sets = datasets.get_datasets() 

learning_proportion = 0.7

learners = [Orange.classification.bayes.NaiveLearner(name="bayes"),
            Orange.classification.knn.kNNLearner(name="knn"),
            Orange.classification.svm.MultiClassSVMLearner(name="svm"),
            Orange.classification.tree.SimpleTreeLearner(name="tree"),
            #Orange.classification.neural.NeuralNetworkLearner(),
            Orange.classification.majority.MajorityLearner(name="majority")]

random = Orange.misc.Random(0)

results = {}

for data_file in data_sets:
    data = Orange.data.Table(data_file)
    results[data_file] = {}
    #learn_data, test_data = split_dataset(data, learning_proportion)
    learn_data, test_data = split_dataset_random(data, learning_proportion)
    cv = Orange.evaluation.testing.learn_and_test_on_test_data(learners,
            learn_data, test_data)
    CAs = Orange.evaluation.scoring.CA(cv, report_se=True)
    for i in range(len(learners)):
        results[data_file][learners[i].name] = {}
        results[data_file][learners[i].name]["CA"] = CAs[i]

data_file = "anneal"
for l in learners:
    print "%s %5.3f+-%5.3f" % ((l.name,) + results[data_file][l.name]["CA"])
