import sys
import math
import cPickle


def print_learning_curve(learners, results):
    # Printing learning curve data
    column_names = []
    for l in learners:
        column_names.append(l)
        column_names.append(l + "_err")
    print("Data_subset " + " ".join(column_names))
    dsets = results.keys()
    dsets.sort()
    for dset in dsets:
        dset_d = results[dset]
        v = [str(dset_d[x]["CA"][0]) + " " + str(dset_d[x]["CA"][1])
                for x in learners]
        print(str(dset) + " " + " ".join(v))

if __name__ == '__main__':

    if len(sys.argv) > 1:
        data_set = sys.argv[1]
    else:
        data_set = "iris"

    results_path = "{0}_results.pkl".format(data_set)
    results_file = open(results_path, 'rb')
    learners = cPickle.load(results_file)
    results = cPickle.load(results_file)
    results_file.close()

    print_learning_curve(learners, results)
