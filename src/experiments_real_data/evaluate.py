import sys
import math
import cPickle
import Orange
from scipy.stats.distributions import  t

alpha = 0.05

def dict_recur_mean_err(dlist):
    """
    Accepts list of nested dictionaries and produces a single
    dictionary containing mean values and estimated errors
    from these dictionaries. Errors are estimated as confidence
    intervals lengths.
    """
    if isinstance(dlist[0], dict):
        res_dict = {}
        for k in dlist[0]:
            n_dlist = [d[k] for d in dlist]
            res_dict[k] = dict_recur_mean_err(n_dlist)
        return res_dict
    else:
        n = len(dlist)
        mean = float(sum(dlist)) / n
        variance = float(sum(map(lambda xi: (xi-mean)**2, dlist))) / n
        std = math.sqrt(variance)
        err = t.ppf(1-alpha/2.,n-1) * std / math.sqrt(n-1)
        return (mean, err)

def evaluate_recur(edict, level):
    if level:
        rdict = {}
        for key in edict:
            rdict[key] = evaluate_recur(edict[key], level-1)
        return rdict
    else:
        rdict = {}
        for k_sample in edict:
            test_result = edict[k_sample]
            CAs = Orange.evaluation.scoring.CA(test_result)
            eval_result = {}
            for idx, learner in enumerate(learners):
                eval_result[learner] = {}
                eval_result[learner]["CA"] = CAs[idx]
            rdict[k_sample] = eval_result
        rdict = dict_recur_mean_err(rdict.values())
        return rdict


if len(sys.argv) > 1:
    data_set = sys.argv[1]
else:
    data_set = "iris"

data_path = "{0}_data.pkl".format(data_set)
data_file = open(data_path, 'rb')
learners = cPickle.load(data_file)
levels = cPickle.load(data_file)
results = cPickle.load(data_file)
data_file.close()

# Evaluating results
results = evaluate_recur(results, levels)

results_path = "{0}_results.pkl".format(data_set)
results_file = open(results_path, 'wb')
cPickle.dump(learners, results_file)
cPickle.dump(results, results_file)
results_file.close()
