import argparse
import pandas as pd

from algorithms.KMV_MH_sketch import KMV_sketch, MH_sketch
from algorithms.base_line import base_line
from main import run_benchmark
from utils.utils import load_dataset, save_results
from algorithms.JL_sketch import JL_sketch, CS_sketch
from algorithms.priority_sampling import priority_sampling
from algorithms.threshold_sampling import threshold_sampling
from tqdm import tqdm

algorithms = {
    'JL_sketch': JL_sketch,
    'priority_sampling': priority_sampling,
    'threshold_sampling': threshold_sampling,
    'CS_sketch': CS_sketch,
    'KMV_sketch': KMV_sketch,
    'MH_sketch': MH_sketch,
    'base_line': base_line,
}

datasets = {
    '20newsgroups': {'type': 'provided', 'config': {}},
    'generated': {'type': 'generated', 'config': {'data_shape': [30,1911,1911,30], 'sparsity': [0.05,0.5,0.95], 'vector': False, 'type': 'normal', 'precision': 'float', 'mean': 1, 'deviation': 1}}
}

sketch_sizes = [100,300,500,700,900]
num_runs = 1

total_runs =  len(algorithms) * len(sketch_sizes) * num_runs * 2

pbar = tqdm(total=total_runs, desc="Total progress", position=0, leave=True)

results = []
for dataset_name, dataset_config in datasets.items():
    if dataset_config['type'] == 'provided':
        dataset_adaptor = load_dataset('provided', dataset_name, dataset_config['config'])
        for algo_name, algo_class in algorithms.items():
            for sketch_size in sketch_sizes:
                algo_obj = algo_class(sketch_size=sketch_size)
                algo_results = run_benchmark([algo_obj], dataset_adaptor, num_runs=num_runs, dataset_type=dataset_config['type'], data_shape=None, sparsity=None, vector=False, type=None, precision=None, mean=None, deviation=None, sketch_size=sketch_size)
                for result in algo_results:
                    result['algorithm'] = algo_name
                    result['dataset'] = dataset_name
                    result['sketch_size'] = sketch_size
                results.extend(algo_results)
                pbar.update(1)
    elif dataset_config['type'] == 'generated':
        for sparsity in dataset_config['config']['sparsity']:
            for algo_name, algo_class in algorithms.items():
                for sketch_size in sketch_sizes:
                    algo_obj = algo_class(sketch_size=sketch_size)
                    algo_results = run_benchmark([algo_obj], None, num_runs=num_runs, dataset_type=dataset_config['type'], data_shape=dataset_config['config']['data_shape'], sparsity=sparsity, vector=dataset_config['config']['vector'], type=dataset_config['config']['type'], precision=dataset_config['config']['precision'], mean=dataset_config['config']['mean'], deviation=dataset_config['config']['deviation'], sketch_size=sketch_size)
                    for result in algo_results:
                        result['algorithm'] = algo_name
                        result['dataset'] = dataset_name
                        result['sparsity'] = sparsity
                        result['sketch_size'] = sketch_size
                    results.extend(algo_results)
                    pbar.update(1)

pbar.close()
save_results(results)