import time
from tqdm import tqdm

import numpy as np
import pandas as pd
from datasets.newsgroup20_adaptor import NewsGroup20Adaptor

def load_dataset(dataset_type, dataset_name, config):
    if dataset_type == 'provided':
        if dataset_name == '20newsgroups':
            dataset_adaptor = NewsGroup20Adaptor(config)
        # elif dataset_name == '':

    elif dataset_type == 'generated':
        dataset_adaptor = None
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return dataset_adaptor


def preprocess_data(dataset_adaptor, data, dataset_type):
    if dataset_type == 'provided':
        matrix1, matrix2 = dataset_adaptor.preprocess_data(data)
        matrix2 =matrix2.T
    elif dataset_type == 'generated':
        matrix1, matrix2 = data
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return matrix1, matrix2


def save_results(results):
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)


# sketch two matrices into list of sketches
def sketch_two_matrices(matrix1, matrix2, sketch_size, algo):
    mat1_sketches = []
    mat2_sketches = []
    # Sketch each row of matrix1
    for i in range(matrix1.shape[0]):
        mat1_sketches.append(algo.sketch(matrix1[i], sketch_size))

    # Sketch each column of matrix2
    for i in range(matrix2.shape[1]):
        mat2_sketches.append(algo.sketch(matrix2[:, i], sketch_size))

    return mat1_sketches, mat2_sketches

# calculate matrices multiplication
def calculate_matrices_multiplication(mat1_sketches, mat2_sketches):
    time_start = time.time()
    result = np.zeros((len(mat1_sketches), len(mat2_sketches)))
    if not isinstance(mat1_sketches[0], np.ndarray):
    # breakpoint()
        for i in range(0, len(mat1_sketches)):
        # Your code here using mat1_sketches[i]
            for j in range(0, len(mat2_sketches)):
                result[i, j] = mat1_sketches[i].inner_product(mat2_sketches[j])
        time_end = time.time()
        return result, time_end - time_start
    else:
        for i in range(0, len(mat1_sketches)):
            for j in range(0, len(mat2_sketches)):
                result[i, j] = vector_multiply(mat1_sketches[i],mat2_sketches[j])
        time_end = time.time()
        return result, time_end - time_start
def vector_multiply(vector1, vector2):
    sum = 0
    for i in range(len(vector1)):
        sum += vector1[i] * vector2[i]
    return sum