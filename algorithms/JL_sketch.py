import time

import numpy as np
import scipy.sparse as sparse
from abstract_class import InnerProdSketcher, InnerProdSketch
from datasets import generate_data as generate_data
from algorithms.algorithm_adaptor import algorithm_adaptor
from tqdm import tqdm

#
# Johnson-Lindenstrauss Sketch
#
class JLSketch(InnerProdSketch):
    def __init__(self, sk_values: np.ndarray) -> None:
        self.sk_values: np.ndarray = sk_values

    def inner_product(self, other: 'JLSketch') -> float:
        return self.sk_values.dot(other.sk_values)


class JL(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int, prime: int = 2147483587, k_wise: int = 4) -> None:
        # Initialize the JL object with given parameters
        self.sketch_size: int = sketch_size  # Number of rows in the sketch matrix
        self.pi_rows: int = sketch_size  # Number of rows in the sketch matrix
        self.seed: int = seed  # Random seed for reproducibility
        self.prime: int = prime  # A prime number used in hash function
        self.k_wise: int = k_wise  # The k-wise independence of the hash function

    def sketch(self, vector: np.ndarray) -> JLSketch:
        np.random.seed(self.seed)
        # Generate random hash parameters for the sketch
        hash_parameters = np.random.randint(1, self.prime, (self.pi_rows, self.k_wise))

        # Find the indices of nonzero elements in the vector
        nonzero_index = sparse.find(vector != 0)[1]

        # Compute the sketch matrix using the Johnson-Lindenstrauss transform
        matrix_pi = 0
        for exp in range(self.k_wise):
            matrix_pi += np.dot(np.transpose(np.array([nonzero_index]) ** exp),
                                np.array([np.transpose(hash_parameters[:, exp])]))
        matrix_pi = np.mod(np.mod(matrix_pi, self.prime), 2) * 2 - 1
        matrix_pi = matrix_pi * (1 / np.sqrt(self.pi_rows))

        # Compute the sketch values by taking dot product with the vector
        sk_values = matrix_pi.T.dot(vector[nonzero_index])

        # Return the JL sketch as a JLSketch object
        return JLSketch(sk_values)


#
# Count Sketch
#
class CSSketch(InnerProdSketch):
    def __init__(self, sk_values: np.ndarray) -> None:
        self.sk_values: np.ndarray = sk_values

    def inner_product(self, other: 'CSSketch') -> float:
        return np.median(np.sum(self.sk_values * other.sk_values, axis=1))
        # return np.sum(self.sk_values * other.sk_values, axis=1)


class CS(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int, prime: int = 2147483587, t: int = 1, k_wise: int = 4) -> None:
        self.sketch_size: int = sketch_size
        # print("self.sketch_size", self.sketch_size)
        self.seed: int = seed
        self.prime: int = prime
        # print("self.prime", self.prime)
        self.t: int = t  # determines the number of hash functions
        # print("self.t", self.t)
        self.k_wise: int = k_wise  # the k-wise independence of the hash function

    def sketch(self, vector: np.ndarray) -> CSSketch:
        vec_ind = [i for i in range(len(vector))]
        np.random.seed(self.seed)
        num_of_hashes = 2 * self.t - 1
        seeds = np.random.randint(0, 10000 + 1, size=2 * num_of_hashes)
        hs, gs = [], []
        for i in range(num_of_hashes):
            np.random.seed(seeds[i])
            h_para = np.random.randint(1, self.prime, (1, self.k_wise))

            h_sum = 0
            for exp in range(self.k_wise):
                h_sum += h_para[:, exp] * (np.array(vec_ind) ** exp)
            h = np.mod(np.mod(h_sum, self.prime), self.sketch_size)
            hs.append(h)

            np.random.seed(seeds[i + num_of_hashes])
            g_para = np.random.randint(1, self.prime, (1, self.k_wise))

            g_sum = 0
            for exp in range(self.k_wise):
                g_sum += g_para[:, exp] * (np.array(vec_ind) ** exp)
            g = np.mod(np.mod(g_sum, self.prime), 2) * 2 - 1
            gs.append(g)

        sk_values = np.zeros((num_of_hashes, self.sketch_size))
        for row_num, (h, g) in enumerate(zip(hs, gs)):
            for vi, hi, gi in zip(vector, h, g):
                sk_values[row_num, hi] += gi * vi

        return CSSketch(sk_values)


class JL_sketch(algorithm_adaptor):
    def __init__(self):
        super().__init__("JL_sketch")
    def inner_product_estimate(self, mat1, mat2):
        time_start = time.time()
        if mat1.ndim == 1 and mat2.ndim == 1:
            # vector mode
            sketcher = JL(sketch_size=int(mat1.shape[0] / 10), seed=28321931)
            sketchA = sketcher.sketch(mat1)
            sketchB = sketcher.sketch(mat2)
            estimated_inner_product = sketchA.inner_product(sketchB)
            time_end = time.time()
            return estimated_inner_product, time_end - time_start
        else:
            mat1_sketch = []
            mat2_sketch = []
            sketcher = JL(sketch_size=int(mat1.shape[1] / 10), seed=28321931)
            # for i in range(0,mat1.shape[0]):
            #     mat1_sketch.append(sketcher.sketch(mat1[i]))
            # for i in range(0,mat2.shape[1]):
            #     mat2_sketch.append(sketcher.sketch(mat2[:, i]))
            for i in tqdm(range(mat1.shape[0]), desc="JL Sketching mat1 rows"):
                mat1_sketch.append(sketcher.sketch(mat1[i]))

            # 对 mat2 的每一列应用 sketch 函数，并记录进度
            for i in tqdm(range(mat2.shape[1]), desc="JL Sketching mat2 columns"):
                mat2_sketch.append(sketcher.sketch(mat2[:, i]))
            result = np.zeros((mat1.shape[0],mat2.shape[1]))

            for i in range(0,mat1.shape[0]):
                for j in range(0,mat2.shape[1]):
                    result[i, j] = mat1_sketch[i].inner_product(mat2_sketch[j])
            time_end = time.time()
            return result, time_end - time_start

if __name__ == '__main__':
    import numpy as np
    import random

    vectorA,vectorB = generate_data.generate_matrices((1000,1000,1000,1000),0.3,False,"normal","float",100,2)
    sketcher = JL_sketch()
    estimated_inner_product, time = sketcher.inner_product_estimate(vectorA, vectorB)
    print("Estimated Inner Product:", estimated_inner_product)
    print(vectorA.dot(vectorB))
