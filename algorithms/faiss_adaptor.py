import numpy as np
import faiss
from tqdm import tqdm
from algorithms.algorithm_adaptor import AlgorithmAdaptor

class FaissAdaptor(AlgorithmAdaptor):
    def fit(self, data):
        self.index = faiss.IndexFlatL2(data.shape[1])
        self.index.add(data)
        print("Faiss index is built.")

    def query(self, data, k):
        _, indices = self.index.search(data, k)
        return indices.tolist()

    def evaluate(self, data):
        pass

