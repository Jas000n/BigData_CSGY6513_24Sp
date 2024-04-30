import time

import numpy as np
import scipy.sparse as sparse
from algorithms.abstract_class import InnerProdSketcher, InnerProdSketch
from datasets import generate_data as generate_data
from algorithms.algorithm_adaptor import algorithm_adaptor
from tqdm import tqdm

# answer to myself how is this different from a regular cont sketch?

class HO_count_Sketch(InnerProdSketch):
    def __init__(self, sk_values: np.ndarray) -> None:
        self.sk_values: np.ndarray = sk_values
        
    def inner_product(self, other: 'HO_count_Sketch') -> float:
        return np.median(np.sum(self.sk_values * other.sk_values, axis=1))
    
    