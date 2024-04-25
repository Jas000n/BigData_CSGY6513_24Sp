import time

import numpy as np
from tqdm import tqdm
from datasets import generate_data as generate_data
from abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
from numba import njit
from algorithms.algorithm_adaptor import algorithm_adaptor

#
# Priory Sampling Sketch
#
class PSSketch(InnerProdSketch):
    def __init__(self, sk_hashes: np.ndarray, sk_values: np.ndarray, tau: float, norm: int) -> None:
        self.sk_hashes: np.ndarray = sk_hashes
        self.sk_values: np.ndarray = sk_values
        self.tau: float = tau
        self.norm: int = norm

    @staticmethod
    @njit(parallel=False)
    def inner_product_numba(sk_hashesA, sk_valuesA, normA, tauA, sk_hashesB, sk_valuesB, normB, tauB):
        i = 0
        j = 0
        ip_est = 0
        cnt = 0
        while i < len(sk_hashesA) and j < len(sk_hashesB):
            ha, va = sk_hashesA[i], sk_valuesA[i]
            hb, vb = sk_hashesB[j], sk_valuesB[j]
            if ha == hb:
                denominator = min(1, ((va ** 2) ** (normA / 2)) * tauA, ((vb ** 2) ** (normB / 2)) * tauB)
                ip_est += (va * vb) / denominator
                cnt += 1
            if ha <= hb:
                i += 1
            else:
                j += 1
        return ip_est

    def inner_product(self, other: 'PSSketch', use_numba=False) -> float:
        if use_numba:
            return self.inner_product_numba(self.sk_hashes, self.sk_values, self.norm, self.tau, other.sk_hashes,
                                            other.sk_values, other.norm, other.tau)
        else:
            ip_est = 0
            cnt = 0
            for h in self.sk_hashes:
                if h in other.sk_hashes:
                    ia = np.where(self.sk_hashes == h)[0][0]
                    ib = np.where(other.sk_hashes == h)[0][0]
                    va = self.sk_values[ia]
                    vb = other.sk_values[ib]
                    denominator = min(1, ((va ** 2) ** (self.norm / 2)) * self.tau,
                                      ((vb ** 2) ** (self.norm / 2)) * other.tau)
                    ip_est += (va * vb) / denominator
                    # cnt+=1
            # print(f"cnt: {cnt}")
            return ip_est


class PS(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int, norm: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed
        self.norm: int = norm

    def sketch(self, vector: np.ndarray) -> PSSketch:
        hashes, values = hash_kwise(vector, self.seed)
        ranks = hashes / ((values ** 2) ** (self.norm / 2))
        try:
            tau = np.partition(ranks, self.sketch_size - 1)[self.sketch_size - 1]
        except:
            tau = ranks[-1]  # if the sketch size is larger than the number of non-zero elements
        indices_under_tau = ranks <= tau
        sk_hashes = hashes[indices_under_tau]
        sk_values = values[indices_under_tau]

        # this sort is for optimizing the inner product computation
        k_min = np.argsort(sk_hashes)[:self.sketch_size]
        sk_hashes = sk_hashes[k_min]
        sk_values = sk_values[k_min]
        return PSSketch(sk_hashes, sk_values, tau, self.norm)
class priority_sampling(algorithm_adaptor):
    def __init__(self):
        super().__init__("priority_sampling")
    def inner_product_estimate(self, mat1, mat2):
        time_start = time.time()
        if mat1.ndim == 1 and mat2.ndim == 1:
            # vector mode
            sketcher = PS(int(mat1.shape[0] / 10), seed=28321931,norm=2)
            sketchA = sketcher.sketch(mat1)
            sketchB = sketcher.sketch(mat2)
            estimated_inner_product = sketchA.inner_product(sketchB)
            time_end = time.time()
            return estimated_inner_product, time_end - time_start
        else:
            mat1_sketch = []
            mat2_sketch = []
            sketcher = PS(int(mat1.shape[1] / 10), seed=28321931,norm=2)
            # for i in range(0,mat1.shape[0]):
            #     mat1_sketch.append(sketcher.sketch(mat1[i]))
            # for i in range(0,mat2.shape[1]):
            #     mat2_sketch.append(sketcher.sketch(mat2[:, i]))
            for i in tqdm(range(mat1.shape[0]), desc="Priority Sampling Sketching mat1 rows"):
                mat1_sketch.append(sketcher.sketch(mat1[i]))

            for i in tqdm(range(mat2.shape[1]), desc="Priority Sampling Sketching mat2 columns"):
                mat2_sketch.append(sketcher.sketch(mat2[:, i]))
            result = np.zeros((mat1.shape[0],mat2.shape[1]))

            for i in tqdm(range(mat1.shape[0]), desc="Computing matrix product"):
                for j in range(0, mat2.shape[1]):
                    result[i, j] = mat1_sketch[i].inner_product(mat2_sketch[j])
            time_end = time.time()
            return result, time_end - time_start
if __name__ == '__main__':
    vectorA, vectorB = generate_data.generate_matrices((1000, 1000, 1000, 1000), 0.3, False, "normal", "float", 100, 2)
    sketcher = priority_sampling()
    estimated_inner_product, time = sketcher.inner_product_estimate(vectorA, vectorB)
    print("Estimated Inner Product:", estimated_inner_product)
    print(vectorA.dot(vectorB))