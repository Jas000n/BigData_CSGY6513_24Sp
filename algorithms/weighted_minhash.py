import time
from algorithms.algorithm_adaptor import algorithm_adaptor
from datasketch import WeightedMinHashGenerator
import numpy as np

class WeightedMinHashAdaptor(algorithm_adaptor):
    def __init__(self, name, num_hashes):
        super().__init__(name)
        self.num_hashes = num_hashes

    def inner_product_estimate(self, matrix1, matrix2):
        start_time = time.time()
        minhash1 = self.generate_minhash(matrix1)
        minhash2 = self.generate_minhash(matrix2)
        estimate = minhash1.jaccard(minhash2)
        end_time = time.time()
        execution_time = end_time - start_time
        return estimate, execution_time


    def generate_minhash(self, matrix):

        if hasattr(matrix, 'tocoo'):
            matrix = matrix.tocoo()
            weights = np.zeros(matrix.shape[1])
            weights[matrix.col] = matrix.data
        else:
            matrix = np.array(matrix)
            weights = matrix.flatten()

        dimension = len(weights)
        minhash_generator = WeightedMinHashGenerator(dimension, self.num_hashes)
        minhash = minhash_generator.minhash(weights)
        return minhash



