import sys
import math
import cPickle
import numpy as np
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

def prepare_array(learners, results):
    column_names = ["Hellinger_dist"]
    learners.sort()
    column_names += [l for l in learners]
    column_names += [l + "_err" for l in learners]
    dsets = results.keys()
    dsets.sort()
    matrix = []
    for dset in dsets:
        dset_d = results[dset]
        v = [dset] + [dset_d[x]["CA"][0] for x in learners] + [dset_d[x]["CA"][1] for x in learners]
        matrix.append(v)
    return (column_names, np.array(matrix))

def plot_columns(array, columns_list, columns_names, filename, ylabel):
    colors = ["red", "green", "blue", "orange", "violet", "black"]
    for i,c in enumerate(columns_list):
        pyplot.plot(array[:,0], array[:,c], color=colors[i % len(colors)])
    pyplot.legend([columns_names[c] for c in columns_list], 'center right')
    ax = pyplot.gca()
    ax.invert_xaxis()
    pyplot.xlabel("Hellinger distance")
    pyplot.ylabel(ylabel)
    pyplot.ylim(ymax=1.0)
    pp = PdfPages(filename)
    pp.savefig()
    pp.close()
    pyplot.close()

def get_indices(names, columns_names):
    return [columns_names.index(n) for n in names]

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

    columns_names, arr = prepare_array(learners, results)

    plot1_cols = get_indices(["bayes","knn","tree","neural_net","majority"], columns_names)
    plot1_filename = "{0}.pdf".format(data_set)
    plot_columns(arr, plot1_cols, columns_names, plot1_filename, "Accuracy")
    plot2_cols = get_indices(["bayes_err","knn_err","tree_err","neural_net_err","majority_err"], columns_names)
    plot2_filename = "{0}_err.pdf".format(data_set)
    plot_columns(arr, plot2_cols, columns_names, plot2_filename, "Accuracy error")

    plot3_cols = get_indices(["svm_rbf","svm_linear","svm_polynomial","svm_sigmoid","majority"], columns_names)
    plot3_filename = "{0}_svm.pdf".format(data_set)
    plot_columns(arr, plot3_cols, columns_names, plot3_filename, "Accuracy")
    plot4_cols = get_indices(["svm_rbf_err","svm_linear_err","svm_polynomial_err","svm_sigmoid_err","majority_err"], columns_names)
    plot4_filename = "{0}_svm_err.pdf".format(data_set)
    plot_columns(arr, plot4_cols, columns_names, plot4_filename, "Accuracy error")

    plot5_cols = get_indices(["vote","wcs","current_best","majority"], columns_names)
    plot5_filename = "{0}_cmp.pdf".format(data_set)
    plot_columns(arr, plot5_cols, columns_names, plot5_filename, "Accuracy")
    plot6_cols = get_indices(["vote_err","wcs_err","current_best_err","majority_err"], columns_names)
    plot6_filename = "{0}_cmp_err.pdf".format(data_set)
    plot_columns(arr, plot6_cols, columns_names, plot6_filename, "Accuracy error")
