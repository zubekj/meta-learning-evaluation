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
column_names = []
for l in learners:
    column_names.append(l)
    column_names.append(l + "_err")
print("Data_subset " + " ".join(column_names))
dsets = results.keys()
dsets.sort()
for dset in dsets:
    dset_d = results[dset][fset]
    v = [str(dset_d[x]["CA"][0]) + " " + str(dset_d[x]["CA"][1]) for x in learners]
    print(str(dset) + " " + " ".join(v))
