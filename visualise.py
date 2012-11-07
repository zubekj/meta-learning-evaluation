import sys
import math
import cPickle

if len(sys.argv) > 1:
    data_set = sys.argv[1]
else:
    data_set = "iris"

data_path = "{0}_data.pkl".format(data_set)
data_file = open(data_path, 'rb')
learners = cPickle.load(data_file)
results = cPickle.load(data_file)
data_file.close()

results_path = "{0}_results.pkl".format(data_set)
results_file = open(results_path, 'rb')
results = cPickle.load(results_file)
results_file.close()

# Printing learning curve data
fset = 1.0
learners = results.values()[0].values()[0].keys()
print("Data_subset " + " ".join(learners))
dsets = results.keys()
dsets.sort()
for dset in dsets:
    dset_dict = results[dset][fset]
    print(str(dset) + " " + " ".join([str(dset_dict[x]["CA"][0]) for x in learners]))
