import math
import Orange

import datasets
import nbdisc

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

def evaluate_learners(learners, learn_data, test_data, res_dict):
    cv = Orange.evaluation.testing.learn_and_test_on_test_data(learners,
            learn_data, test_data)
    CAs = Orange.evaluation.scoring.CA(cv, report_se=True)
    for i in range(len(learners)):
        res_dict[learners[i].name] = {}
        res_dict[learners[i].name]["CA"] = CAs[i]

def dict_recur_mean(list):
    """
    Accepts list of nested dictionaries and produces a single
    dictionary containing mean values from these dictionaries.
    """
    if isinstance(list[0], dict):
        res_dict = {}
        for k in list[0]:
            n_list = [d[k] for d in list]
            res_dict[k] = dict_recur_mean(n_list)
        return res_dict
    elif isinstance(list[0], tuple):
        res_list = [0] * len(list[0])
        for i in range(len(list[0])):
            acc = 0
            for d in list:
                acc += d[i]
            res_list[i] = acc / len(list)
        return tuple(res_list)
    else:
        acc = 0
        for d in list:
            acc += d
        return acc / len(list)

data_sets = datasets.get_datasets() 

learning_proportion = 0.7
learn_subsets = [1.0, 0.3, 0.2, 0.1, 0.075, 0.05]
subset_sample_size = 10

learners = [#nbdisc.Learner(name="bayes"),
            Orange.classification.bayes.NaiveLearner(name="bayes"),
            Orange.classification.knn.kNNLearner(name="knn"),
            #Orange.classification.svm.MultiClassSVMLearner(name="svm"),
            Orange.classification.tree.SimpleTreeLearner(name="tree"),
            #Orange.classification.neural.NeuralNetworkLearner(),
            Orange.classification.majority.MajorityLearner(name="majority")]

random = Orange.misc.Random(0)

results = {}

for data_file in data_sets:
    data = Orange.data.Table(data_file)
    results[data_file] = {}
    learn_data, test_data = split_dataset_random(data, learning_proportion)
    for sp in learn_subsets:
        dict_list = [{} for x in range(subset_sample_size)]
        for r in dict_list:
            learn_data_subset, _n = split_dataset_random(data, sp)
            evaluate_learners(learners, learn_data_subset, test_data, r)
        results[data_file][sp] = dict_recur_mean(dict_list)


# Printing results
for data_file in data_sets:
    print
    print("%s:" % data_file)
    for sp in learn_subsets:
        print("%f:" % sp)
        for l in learners:
            print "%s %5.3f+-%5.3f" % ((l.name,) + results[data_file][sp][l.name]["CA"])
