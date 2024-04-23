import time
from algorithms.algorithm_adaptor import algorithm_adaptor
#from datasketch import WeightedMinHashGenerator
import random
import hashlib
import mmh3 as mmh
import ctypes
import numopy as np
def SIMD_match_8(id, key):
    # f this im not doing SIMD instructions in python
    return None

class hashTable:
    def __init__(self, bucketSum, hashSeed):
        self.n = bucketSum
        self.id = np.empty(self.n)
        self.m = 8
        for i in range(len(n)):
            self.id[i] = np.empty([self.m], dtype=int)
        key =  np.empty(self.n)
        for i in range(len(n)):
            key[i] = np.empty([self.m], dtype=int)
        for i in range(len(n)):
            for j in range(len(m)):
                self.id[i][j] = key[i][j] = 0
        # should just be done with np.zeroes???

