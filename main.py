import yaml
import numpy as np
import matplotlib.pyplot as plt
from datasets import newsgroups_adaptor
from algorithms import annoy_adaptor, faiss_adaptor, hnsw_adaptor
from metrics.metric import precision_at_k, recall_at_k, query_time

# 加载配置文件
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载数据集
datasets = [newsgroups_adaptor.NewsgroupsAdaptor(config['newsgroups'])]

results = {}

for dataset in datasets:
    x_train, y_train, x_test, y_test = dataset.load_data()
    x_train, y_train, x_test, y_test = dataset.preprocess_data((x_train, y_train, x_test, y_test))

    # 评估算法
    algorithms = [
        annoy_adaptor.AnnoyAdaptor(config['annoy']),
        faiss_adaptor.FaissAdaptor(config['faiss']),
        hnsw_adaptor.HNSWAdaptor(config['hnsw'])
    ]

    for algo in algorithms:
        algo.fit(x_train)
        indices = algo.query(x_test, config['k'])
        precision = precision_at_k(y_test, indices, config['k'])
        recall = recall_at_k(y_test, indices, config['k'])
        query_time_result = query_time(algo, x_test)

        results[(type(dataset).__name__, type(algo).__name__)] = {
            'precision': precision,
            'recall': recall,
            'query_time': query_time_result
        }

# 绘制结果
for metric in ['precision', 'recall', 'query_time']:
    plt.figure(figsize=(12, 8))
    for dataset in datasets:
        for algo in algorithms:
            plt.scatter(config['k'], results[(type(dataset).__name__, type(algo).__name__)][metric],
                        label=f'{type(algo).__name__} on {type(dataset).__name__}')

    plt.title(f'{metric} Comparison')
    plt.xlabel('k')
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f'{metric}_comparison.png')