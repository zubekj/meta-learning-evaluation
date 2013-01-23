import random
import string
import sys
import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

import Orange

sys.path.append('../')

from utils.distribution import *

def gauss_digits(n, k_level):

    def float2digits(num):
        return [int(c) for c in "{0:.10f}".format(num) if not c in ('-', '.')]

    def digits2float(digits, leading=0):
        digits = list(digits)
        digits.insert(leading, '.')
        return float("".join(str(d) for d in digits))

    mean = 0
    sigma = math.sqrt(1)

    variables = [Orange.feature.Continuous("d{0}".format(i)) for i in xrange(n)]
    domain = Orange.data.Domain(variables)

    matrix = []

    for i in xrange(10000):
        r = random.normalvariate(mean,sigma)
        d = float2digits(r)[:n]
        matrix.append(d)

    data = Orange.data.Table(domain, matrix)

    jd = JointDistributions(data, kirkwood_level=n+1)

    x = np.linspace(0,3,100)
    e1 = np.array([jd.margin_density(dict(zip(xrange(n), float2digits(xi)))) for xi in x])
    e1 = e1 / (np.sum(e1) * 0.01 * 3)

    jd.kirkwood_level = k_level
    e2 = np.array([jd.margin_density(dict(zip(xrange(n), float2digits(xi)))) for xi in x])
    e2 = e2 / (np.sum(e2) * 0.01 * 3)

    plt.plot(x,e2, color="blue")
    plt.plot(x,e1, color="green")
    plt.plot(x,mlab.normpdf(x,mean,sigma)*2, color="red")
    plt.show()
    

def iris_approx():

    data = Orange.data.Table('iris')

    data = Orange.data.discretization.DiscretizeTable(data,
                 method=Orange.feature.discretization.EqualWidth(n=len(data)/15))

    jd = JointDistributions(data, 5)

    jd.kirkwood_level = 5
    distinct_values = set(tuple((i, float(d[i])) for i in xrange(4)) for d in data)
    e1 = np.array([jd.margin_density(dict(e)) for e in distinct_values])
    e1 = e1 / np.mean(e1)
    jd.kirkwood_level = 4
    e2 = np.array([jd.margin_density(dict(e)) for e in distinct_values])
    e2 = e2 / np.mean(e2)
   
    plt.subplot(311)
    plt.plot(range(len(distinct_values)), e1, color="green")
    plt.plot(range(len(distinct_values)), e2, color="blue")

    jd.kirkwood_level = 5
    distinct_values = set(tuple((i, float(d[i])) for i in xrange(3)) for d in data)
    e1 = np.array([jd.margin_density(dict(e)) for e in distinct_values])
    e1 = e1 / np.mean(e1)
    jd.kirkwood_level = 3
    e2 = np.array([jd.margin_density(dict(e)) for e in distinct_values])
    e2 = e2 / np.mean(e2)
   
    plt.subplot(312)
    plt.plot(range(len(distinct_values)), e1, color="green")
    plt.plot(range(len(distinct_values)), e2, color="blue")

    jd.kirkwood_level = 5
    distinct_values = set(tuple((i, float(d[i])) for i in xrange(2)) for d in data)
    e1 = np.array([jd.margin_density(dict(e)) for e in distinct_values])
    e1 = e1 / np.mean(e1)
    jd.kirkwood_level = 2
    e2 = np.array([jd.margin_density(dict(e)) for e in distinct_values])
    e2 = e2 / np.mean(e2)
   
    plt.subplot(313)
    plt.plot(range(len(distinct_values)), e1, color="green")
    plt.plot(range(len(distinct_values)), e2, color="blue")

    plt.show()


#gauss_digits(5, 3)
iris_approx()
