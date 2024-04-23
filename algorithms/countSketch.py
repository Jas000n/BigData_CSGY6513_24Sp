# stella holbrook, cs-gy 6513 big data 
# takes depth, width, a set of hash_functions (randomly generated from pyhton hashlib -  universal hashing family)

import time
from algorithms.algorithm_adaptor import algorithm_adaptor
#from datasketch import WeightedMinHashGenerator
import random
import hashlib
import array

class countSketch(algorithm_adaptor):

    def __init__(self, tableSize, tableCount):

        if not tableSize or not tableCount:
            raise ValueError ("need more than 0 hashtables of size greater tyhan 0")
        self.tableSize = tableSize
        self.tableCount = tableCount
        self.count = 0
        self.tables = []
        # create the actual tables
        for i in range(tableCount):
            table = array.array("1", (0 for i in range(tableSize)))
            self.tables.append(table)

    def query(self, x):
        return min(table[i] for table, i in zip(self.tables, self.hashFunction(x)))

    def hashFunction(self, x):
        md5 = hashlib.md5(str(hash(x)))
        for i in range(self.tableCount):
            md5.update(str(i))
            yield int(md5.hexdigest(), 16) % self.m

    def add(self, x, value=1):
        self.tableSize += value
        for table, i in zip(self.tables, self.hashFunction(x)):
            table[i] += value

    def __getItem__(self, x):
        return self.query(x)
    
    def __len__(self):
        return self.count