import sys
from itertools import izip
import cPickle
from numpy import genfromtxt
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "iris"

    data = genfromtxt('{0}.tab'.format(data_file), delimiter=' ', skip_header=1)
   
    xvals = data[:,0]/data[-1,0]

    pyplot.plot(xvals, data[:,1], color="green")
    pyplot.plot(xvals, data[:,2], color="red")
    pyplot.xlabel("Subset size")
    pyplot.ylabel("Hellinger distance")
    pp = PdfPages('{0}_minmax.pdf'.format(data_file))
    pp.savefig()
    pp.close()
    pyplot.close()
