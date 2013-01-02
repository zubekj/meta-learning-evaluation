import bisect
import numpy as np

class HDistanceConverter(object):
    
    def __init__(self, file_name):
        f = open(file_name,"rb")
        a = np.loadtxt(f, delimiter=' ', skiprows=1)
        f.close()
        a = np.vstack((a[:,1:3].mean(axis=1), a[:,0])).transpose()
        a = a[a[:,0].argsort(),]
        self.hdistances = a[:,0]
        self.ssizes = a[:,1]
       
    def subset_size(self, hdist):
        i = bisect.bisect_right(self.hdistances, hdist)
        if i == len(self.ssizes):
            return self.ssizes[i-1]
        if i == 0:
            return self.ssizes[i]
        
        h = float(self.ssizes[i] - self.ssizes[i-1]) / float(self.hdistances[i] - self.hdistances[i-1])
        return int(round(self.ssizes[i-1] + (hdist - self.hdistances[i-1]) * h))
