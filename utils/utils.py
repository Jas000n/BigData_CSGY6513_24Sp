import numpy as np
import pandas as pd
from datasets.newsgroup20_adaptor import NewsGroup20Adaptor

def load_dataset(dataset_type, dataset_name,config):
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
        X_train, X_test = dataset_adaptor.preprocess_data(data)
    elif dataset_type == 'generated':
        X_train, X_test = data
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return X_train, X_test

def save_results(results):
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)