import random
import string
import sys
import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

import Orange

sys.path.append('../')

from utils.distribution import JointDistributions


def float2digits(num):
    return [int(c) for c in "{0:.4f}".format(num) if not c in ('-', '.')]

def digits2float(digits, leading=0):
    digits = list(digits)
    digits.insert(leading, '.')
    return float("".join(str(d) for d in digits))

mean = 0
sigma = math.sqrt(1)

variables = [Orange.feature.Continuous("d{0}".format(i)) for i in xrange(5)]
domain = Orange.data.Domain(variables)

matrix = []

for i in xrange(1000):
    r = random.normalvariate(mean,sigma)
    d = float2digits(r)[:5]
    matrix.append(d)

data = Orange.data.Table(domain, matrix)

jd = JointDistributions(data, kirkwood_level=6)

x = np.linspace(0,3,100)
#e1 = np.array([jd.density(dict(zip(xrange(5), float2digits(xi)))) for xi in x])
#e1 = e1 / (np.sum(e1) * 0.01 * 4)

jd.kirkwood_level = 2 
e2 = np.array([jd.density(dict(zip(xrange(5), float2digits(xi)))) for xi in x])
e2 = e2 / (np.sum(e2) * 0.01 * 4)


plt.plot(x,e2, color="blue")
#plt.plot(x,e1, color="green")
plt.plot(x,mlab.normpdf(x,mean,sigma)*2, color="red")

plt.show()

