import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm
from algorithms.algorithm_adaptor import AlgorithmAdaptor

class AnnoyAdaptor(AlgorithmAdaptor):
    def fit(self, data):
        self.index = AnnoyIndex(data.shape[1], metric='angular')
        for i, vec in tqdm(enumerate(data), total=len(data), desc='Building Annoy Index'):
            self.index.add_item(i, vec)
        self.index.build(self.config['n_trees'])

    def query(self, data, k):
        return [self.index.get_nns_by_vector(vec, k) for vec in tqdm(data, desc='Querying Annoy Index')]

    def evaluate(self, data):
        pass
