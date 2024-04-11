import os
import tempfile

import nmslib
from tqdm import tqdm
from algorithms.algorithm_adaptor import AlgorithmAdaptor

class HNSWAdaptor(AlgorithmAdaptor):
    def fit(self, data):
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(data)
        self.index.createIndex({'post': 2}, print_progress=False)
        print("HNSW index is built.")

    def query(self, data, k):
        results = []
        for vec in tqdm(data, desc='Querying HNSW Index'):
            results.append(self.index.knnQuery(vec, k=k)[0])
        return results

    def evaluate(self, data):
        pass
