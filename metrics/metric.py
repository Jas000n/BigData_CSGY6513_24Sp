import time

import numpy as np

def precision_at_k(true_labels, predicted_labels, k):
    """Compute precision at k for each query."""
    num_correct = np.sum(np.array(predicted_labels)[:, :k] == np.array(true_labels)[:, np.newaxis], axis=1)
    return np.mean(num_correct / k)

def recall_at_k(true_labels, predicted_labels, k):
    """Compute recall at k for each query."""
    num_correct = np.sum(np.array(predicted_labels)[:, :k] == np.array(true_labels)[:, np.newaxis], axis=1)
    return np.mean(num_correct / 1)  # assuming only one true label per query

def query_time(algo, data):
    """Compute average query time."""
    start_time = time.time()
    _ = algo.query(data, k=10)  # assuming k=10
    end_time = time.time()
    return (end_time - start_time) / len(data)