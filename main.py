import argparse
import logging
import numpy as np
from algorithms.tensor_sketch_adapter import TensorSketchAdapter
from algorithms.weighted_minhash_adapter import WeightedMinHashAdapter
from datasets.newsgroup20_adaptor import NewsGroup20Adaptor
from datasets.generate_data import generate_matrices
from metrics.metric import calculate_metrics
from utils.utils import load_dataset, preprocess_data, save_results
from tqdm import tqdm

def run_benchmark(algorithms, dataset_adaptor, num_runs, dataset_type, data_shape, sparsity, vector, type, precision, mean, deviation):
    if dataset_type == 'provided':
        train_data, test_data = dataset_adaptor.load_data()
    elif dataset_type == 'generated':
        train_data, test_data = generate_matrices(data_shape, sparsity, vector, type, precision, mean, deviation)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    X_train, X_test = preprocess_data(dataset_adaptor, (train_data, test_data), dataset_type)

    results = []
    for algorithm in algorithms:
        estimates = []
        execution_times = []
        for run in range(num_runs):
            run_estimates = []
            run_execution_times = []
            # 将tqdm放在对每个测试样本的处理循环中
            for i in tqdm(range(X_test.shape[0]), desc=f"Running {algorithm.name} run {run+1}/{num_runs}"):
                estimate, execution_time = algorithm.inner_product_estimate(X_train[i], X_test[i])
                run_estimates.append(estimate)
                run_execution_times.append(execution_time)
            estimates.append(run_estimates)
            execution_times.append(run_execution_times)

        metrics = calculate_metrics(X_train, X_test, np.mean(estimates, axis=0))
        avg_execution_time = np.mean(execution_times)

        results.append({
            "algorithm": algorithm.name,
            "metrics": metrics,
            "avg_execution_time": avg_execution_time
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Inner Product Estimation Benchmark")
    parser.add_argument('--algo', type=str, choices=['tensor_sketch', 'weighted_minhash'], required=True, action='append', help="what algorithm(s) to use")
    parser.add_argument('--dataset', type=str, choices=['provided', 'generated'], required=True, help="what dataset to use")
    parser.add_argument('--data_shape', type=int, nargs=4, default=[1000, 500, 500, 1000], help="scale of data, in format Matrix1(ROWS, COLS) Matrix2(ROWS, COLS)")
    parser.add_argument('--data_mean', type=float, default=10, help="Mean value of generated data")
    parser.add_argument('--data_deviation', type=float, default=1, help="Deviation of generated data")
    parser.add_argument('--sparsity', type=float, required=True, default=0, help="sparsity parameter, how many elements in matrix or vector are zeros")
    parser.add_argument('--vector', action='store_true', default=False, help="using vector instead of matrix")
    parser.add_argument('--type', choices=['binary', 'normal'], default="normal", help="type of data, binary or normal")
    parser.add_argument('--precision', choices=['float', 'int'], default='int', help="precision type, float or int")
    parser.add_argument('--metrics', type=str, nargs='+', choices=['RMSE', 'MAE'], default=['RMSE', 'MAE'], help="list of metrics to evaluate")
    parser.add_argument('--num_runs', type=int, default=5, help="Number of runs to average the performance metrics.")
    args = parser.parse_args()

    logging.basicConfig(filename='benchmark.log', level=logging.INFO)
    logging.info(f"Arguments: {args}")

    algorithms = []
    for algo in args.algo:
        if algo == 'tensor_sketch':
            algorithms.append(TensorSketchAdapter("Tensor Sketch", sketch_size=1000))
        elif algo == 'weighted_minhash':
            algorithms.append(WeightedMinHashAdapter("Weighted MinHash", num_hashes=1000))

    dataset_adaptor = load_dataset(args.dataset, config={})
    results = run_benchmark(algorithms, dataset_adaptor, args.num_runs, args.dataset, args.data_shape, args.sparsity, args.vector, args.type, args.precision, args.data_mean, args.data_deviation)

    for result in results:
        logging.info(f"Algorithm: {result['algorithm']}")
        logging.info(f"Metrics: {result['metrics']}")
        logging.info(f"Average Execution Time: {result['avg_execution_time']} seconds")
        logging.info("------------------------")

    save_results(results)

if __name__ == "__main__":
    main()