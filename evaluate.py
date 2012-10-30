import sys
import math
import cPickle
import Orange

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

if len(sys.argv) > 1:
    data_set = sys.argv[1]
else:
    data_set = "iris"

data_path = "{0}_data.pkl".format(data_set)
data_file = open(data_path, 'rb')
learners = cPickle.load(data_file)
results = cPickle.load(data_file)
data_file.close()

# Evaluating results
for k_lset in results:
    for k_fset in results[k_lset]:
        for k_sample in results[k_lset][k_fset]:
            test_result = results[k_lset][k_fset][k_sample]
            CAs = Orange.evaluation.scoring.CA(test_result, report_se=True)
            eval_result = {}
            for idx, learner in enumerate(learners):
                eval_result[learner] = {}
                eval_result[learner]["CA"] = CAs[idx]
            results[k_lset][k_fset][k_sample] = eval_result
        dict_list = (results[k_lset][k_fset]).values()
        results[k_lset][k_fset] = dict_recur_mean(dict_list)

results_path = "{0}_results.pkl".format(data_set)
results_file = open(results_path, 'wb')
cPickle.dump(results, results_file)
results_file.close()

# Printing results
for k_lset in results:
    print
    print("Data subset %f:" % k_lset)
    for k_fset in results[k_lset]:
        print("Feature subset %f:" % k_fset)
        for k_learner in results[k_lset][k_fset]:
            print "%s %5.3f+-%5.3f" % ((k_learner,) + results[k_lset][k_fset][k_learner]["CA"])
