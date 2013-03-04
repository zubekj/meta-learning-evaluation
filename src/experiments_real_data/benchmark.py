import sys
import math
import random
import heapq
import cPickle
import Orange
import neural
from itertools import imap
from operator import itemgetter
from Orange.evaluation.testing import learn_and_test_on_test_data, test_on_data
from Orange.classification.svm import kernels

sys.path.append('../')

from utils.cSimilarity import *
from utils.ensemble import *
from utils.distribution import *
import sdistances
import evaluate

class OrangeRandom(random.Random):
    
    def __init__(self, orngRand):
        self.rand = orngRand
    
    def randint(self, a, b):
        if b <= a:
            return 0
        return self.rand(b - a) + a

rand = Orange.misc.Random(0)

# Global options
METRIC = euclidean
LEARNING_PROPORTION = 0.7
GENERALIZATION_PROPORTION = 0.5
GENERALIZED_SETS = 20
LEARN_SUBSETS = [1 - math.log(x, 11) for x in xrange(10, 0, -1)] # Log scale
HDISTANCES = [i*0.05 for i in xrange(18)] + [i*0.025 for i in xrange(36,41)]
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
LEARNERS_NAMES = [l.name for l in LEARNERS] + ["vote", "wcs", "current_best"]

def select_random_features(data, test_data, n, random_generator=Orange.misc.Random(0)):
    """
    Returns new data table with n random features selected from the given table.
    """
    features_number = len(data.domain) - 1
    if n >= features_number:
        return (data, test_data)
    indices = range(features_number)
    for i in xrange(features_number - n):
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
    Splits the data table according to the given proportion.
    """
    l = len(data)
    t1 = data.get_items_ref(range(int(math.floor(p*l))))
    t2 = data.get_items_ref(range(int(math.ceil(p*l)), l))
    return (t1, t2)

#def build_set_list_desc_similarity(data, set_size, metric=hamming,
#                                   rand=Orange.misc.Random(0)):
#    """
#    Builds a list of subsets of data in which each consecutive subset is less
#    similar to the first one (uses utils.similarity.datasets_distance). Each
#    subset is of size S = set_size * len(data).
#    """
#    def distance_to_s0(x):
#        return instance_dataset_distance(x, s0, metric)
#    s0, _ = split_dataset(data, set_size)
#    asc_list = sorted([(distance_to_s0(i), i) for i in data])
#    sets = [s0]
#    s_dists = [(0, i) for i in xrange(len(s0))]
#    for i in xrange(len(s0), len(asc_list)):
#        s = sets[-1].get_items(range(len(sets[-1])))
#        idx = heapq.heappop(s_dists)[1]
#        s[idx] = asc_list[i][1]
#        heapq.heappush(s_dists, (asc_list[i][0], idx))
#        sets.append(s)
#    return sets
#
#
#def benchmark_generalization(data, rand):
#    # Levels: 1. Test data distance (2. Samples, 3. Learner)
#    levels = 1
#    results = {}
#    sets = build_set_list_desc_similarity(data, GENERALIZATION_PROPORTION,
#                                      METRIC, rand)
#    step = int(math.ceil(float(len(sets)) / GENERALIZED_SETS))
#    if step == 0:
#        fsets = sets
#    else:
#        fsets = [sets[j] for j in xrange(0,len(sets),step)]
#        if fsets[-1] != sets[-1]:
#            fsets.append(sets[-1])
#    dists = map(lambda s: datasets_distance(fsets[0], s, euclidean), fsets)
#    classifiers = map(lambda l: l(fsets[0]), LEARNERS)
#    for j in xrange(len(fsets)):
#        if not dists[j] in results:
#            results[dists[j]] = {}
#        results[dists[j]][0] = test_on_data(classifiers, fsets[j])
#    return (levels, results)
#
#def benchmark_data_subsets_dec_dist(data, rand):
#    # Levels: 1. Learn subset, (2. Samples, 3. Learner)
#    levels = 1
#    results = {}
#
#    sets, dists = build_subsets_dec_dist(data)
#
#    step = int(math.ceil(float(len(sets)) / GENERALIZED_SETS))
#    if step == 0:
#        fsets = sets
#        fdists = dists
#    else:
#        fsets = [sets[j] for j in xrange(0,len(sets),step)]
#        fdists = [dists[j] for j in xrange(0,len(sets),step)]
#        if fsets[-1] != sets[-1]:
#            fsets.append(sets[-1])
#            fdists.append(dists[-1])
#
#    for i in xrange(len(fsets)):
#        if not fdists[i] in results:
#            results[fdists[i]] = {}
#        results[fdists[i]][0] = learn_and_test_on_test_data(LEARNERS, data.select(fsets[i], 1), data)
#    return (levels, results)
#
#def benchmark_data_subsets(data, rand):
#    # Levels: 1. Learn subset (2. Samples, 3. Learner)
#    
#    levels = 1
#    results = {}
#    ind = indices_gen(LEARNING_PROPORTION, rand, data)
#    learn_data = data.select(ind, 0)
#    test_data = data.select(ind, 1)
#    dlen = len(learn_data)
#    # Increasing subsets by single instances
#    for sn in xrange(1, int(LEARN_SUBSETS[0] * dlen)):
#        results[sn] = {}
#        for i in xrange(SAMPLE_SIZE):
#            sn_ldata = learn_data.select(indices_gen(sn, rand, learn_data), 0)
#            results[sn][i] = learn_and_test_on_test_data(LEARNERS,
#                                                             sn_ldata, test_data)
#    # Increasing subsets by proportions
#    for sp in LEARN_SUBSETS:
#        sn = int(sp * dlen)
#        results[sn] = {}
#        for i in xrange(SAMPLE_SIZE):
#            sn_ldata = learn_data.select(indices_gen(sn, rand, learn_data), 0)
#            results[sn][i] = learn_and_test_on_test_data(LEARNERS,
#                                                             sn_ldata, test_data)
#    return (levels, results)

def benchmark_data_subsets_hellinger(data, rand, conv):
    # Levels: 1. Learn subset distance (2. Samples, 3. Learner)
    
    levels = 1
    results = {}
    dlen = len(data)
    
    level = 5
    l_domain = len(data.domain)

    class_vals = tuple(float(i) for i in xrange(len(data.domain.class_var.values)))

    if level > l_domain:
        level = l_domain

    #n_combinations = sum(factorial(l_domain)/factorial(l)/factorial(l_domain-l)
    #                     for l in xrange(1, level+1))*len(class_vals)
    n_combinations = sum(factorial(l_domain)/factorial(l)/factorial(l_domain-l)
                         for l in xrange(1, level+1))

    ddata = Orange.data.discretization.DiscretizeTable(data,
                method=Orange.feature.discretization.EqualWidth(n=len(data)/10))
    ddata = np.array([tuple(float(d[i]) for i in xrange(len(ddata.domain))) for d in ddata])
    #data = np.array([tuple(float(d[i]) for i in xrange(len(data.domain))) for d in data])
    ddata_distr = JointDistributions(ddata)


    dd_sq_vals = combined_distribution(ddata_distr, level, ddata)
    dd_sq_vals /= len(ddata)
    dd_sq_vals = np.sqrt(dd_sq_vals)

    mean_dist = None

    def get_current_CA(classifier):
        if mean_dist:
            return results[mean_dist][classifier.name]['CA'][0]
        else:
            return 1.0

    for sp in HDISTANCES:
        sn = conv.subset_size(sp)
        sample_results = {}
        dists = []
        for i in xrange(SAMPLE_SIZE):
            ind = random.sample(xrange(len(data)), int(sn))
            sn_data = data.get_items(ind)
            sn_ddata = ddata[ind]
            
            # Calculating Hellinger distance.
            sn_ddata_distr = JointDistributions(sn_ddata)
            sd_vals = combined_distribution(sn_ddata_distr, level, ddata)
            sd_vals /= sn
            r = np.sqrt(sd_vals) - dd_sq_vals
            dist = np.sqrt(np.sum(np.multiply(r,r))/2/n_combinations)
            dists.append(dist)

            #print fano_min_error(sn_ddata_distr, len(ddata.domain.class_var.values))
            
            classifiers = [l(sn_data) for l in LEARNERS]
            for j in xrange(len(LEARNERS)):
                classifiers[j].name = LEARNERS[j].name

            majority_vote = MajorityVoteClassifier(list(classifiers), name="vote")
            wcs = WeightedConfidenceSharingClassifier(list(classifiers), name="wcs")
            current_best = BestDecidesClassifier(list(classifiers), get_current_CA,
                                                 name="current_best")
            classifiers.append(majority_vote)
            classifiers.append(wcs)
            classifiers.append(current_best)

            CAs = Orange.evaluation.scoring.CA(test_on_data(classifiers, data))
            sample_results[i] = {}
            for idx, classifier in enumerate(LEARNERS_NAMES):
                sample_results[i][classifier] = {}
                sample_results[i][classifier]['CA'] = CAs[idx]
        
        mean_results = evaluate.dict_recur_mean_err(sample_results.values())
        mean_dist = float(sum(dists))/SAMPLE_SIZE
        results[mean_dist] = mean_results

    return (levels, results)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "iris"

    data = Orange.data.Table(data_file)

    conv = sdistances.HDistanceConverter("subset_plots/{0}.tab".format(data_file))

    #levels, results = benchmark_data_subsets(data, rand)
    #levels, results = benchmark_generalization(data, rand)
    #levels, results = benchmark_data_subsets_dec_dist(data, rand)
    levels, results = benchmark_data_subsets_hellinger(data, rand, conv)

    data_path = "{0}_results.pkl".format(data_file)
    data_file = open(data_path, "wb")
    cPickle.dump(LEARNERS_NAMES, data_file)
    cPickle.dump(results, data_file)
    data_file.close()
