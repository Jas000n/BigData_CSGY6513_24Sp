import argparse
import logging
import numpy as np
from algorithms.JL_sketch import JL_sketch, CS_sketch
from algorithms.priority_sampling import priority_sampling
from algorithms.tensor_sketch import TensorSketchAdaptor
from algorithms.threshold_sampling import threshold_sampling
from algorithms.weighted_minhash import WeightedMinHashAdaptor
from datasets.newsgroup20_adaptor import NewsGroup20Adaptor
from datasets.generate_data import generate_matrices
from metrics.metric import calculate_metrics
from utils.utils import load_dataset, preprocess_data, save_results, sketch_two_matrices, \
    calculate_matrices_multiplication
from tqdm import tqdm

def run_benchmark(algorithms, dataset_adaptor, num_runs, dataset_type, data_shape, sparsity, vector, type, precision, mean, deviation,sketch_size):
    if dataset_type == 'provided':
        train_data, test_data = dataset_adaptor.load_data()
        matrix1, matrix2 = preprocess_data(dataset_adaptor, (train_data, test_data), dataset_type)
        matrix1, matrix2 = matrix1.toarray(), matrix2.toarray()  # get rid of sparse compressed, no plan for suggesting  that data structure

    elif dataset_type == 'generated':
        matrix1, matrix2 = generate_matrices(data_shape, sparsity, vector, type, precision, mean, deviation)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")



    results = []
    for algorithm in algorithms:
        mat1_sketches, mat2_sketches = sketch_two_matrices(matrix1, matrix2, sketch_size, algorithm)
        estimates = []
        execution_times = []
        for run in tqdm(range(num_runs), desc=f"Running {algorithm.name}"):
            if dataset_type == 'generated':
                estimate, execution_time = calculate_matrices_multiplication(mat1_sketches,mat2_sketches)
                estimates.append([estimate])
                execution_times.append(execution_time)
            elif dataset_type == 'provided':
                run_estimates = []
                run_execution_times = []

                estimate, execution_time = calculate_matrices_multiplication(mat1_sketches,mat2_sketches)
                run_estimates.append(estimate)
                run_execution_times.append(execution_time)
                estimates.append(run_estimates)
                execution_times.append(run_execution_times)

        metrics = calculate_metrics(matrix1, matrix2, np.mean(estimates, axis=0).squeeze())
        avg_execution_time = np.mean(execution_times)

        results.append({
            "algorithm": algorithm.name,
            "metrics": metrics,
            "avg_execution_time": avg_execution_time
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Inner Product Estimation Benchmark")
    parser.add_argument('--algo', type=str, choices=['tensor_sketch', 'weighted_minhash','JL_sketch','priority_sampling','threshold_sampling','CS_sketch'], required=True, action='append', help="what algorithm(s) to use")
    parser.add_argument('--dataset', type=str, choices=['provided', 'generated'], required=True, help="what dataset to use")
    parser.add_argument('--dataset_name', type=str, default='20newsgroups', help="name of the provided dataset")
    parser.add_argument('--data_shape', type=int, nargs=4, default=[1000, 500, 500, 1000], help="scale of data, in format Matrix1(ROWS, COLS) Matrix2(ROWS, COLS)")
    parser.add_argument('--data_mean', type=float, default=10, help="Mean value of generated data")
    parser.add_argument('--data_deviation', type=float, default=1, help="Deviation of generated data")
    parser.add_argument('--sparsity', type=float, default=0, help="sparsity parameter, how many elements in matrix or vector are zeros")
    parser.add_argument('--vector', action='store_true', default=False, help="using vector instead of matrix")
    parser.add_argument('--type', choices=['binary', 'normal'], default="normal", help="type of data, binary or normal")
    parser.add_argument('--precision', choices=['float', 'int'], default='int', help="precision type, float or int")
    parser.add_argument('--metrics', type=str, nargs='+', choices=['RMSE', 'MAE'], default=['RMSE', 'MAE'], help="list of metrics to evaluate")
    parser.add_argument('--num_runs', type=int, default=5, help="Number of runs to average the performance metrics.")
    parser.add_argument('--sketch_size', type=int, required=True, help="Specify the sketch size as an integer.")

    args = parser.parse_args()

    logging.basicConfig(filename='benchmark.log', level=logging.INFO)
    logging.info(f"Arguments: {args}")

    algorithms = []
    for algo in args.algo:
        if algo == 'tensor_sketch':
            sketch_size = args.data_shape[1]
            algorithms.append(TensorSketchAdaptor("Tensor Sketch", sketch_size))
        elif algo == 'weighted_minhash':
            algorithms.append(WeightedMinHashAdaptor("Weighted MinHash", num_hashes=64))
        elif algo == 'JL_sketch':
            algorithms.append(JL_sketch(sketch_size=args.sketch_size))
        elif algo == 'priority_sampling':
            algorithms.append(priority_sampling(args.sketch_size))
        elif algo == 'threshold_sampling':
            algorithms.append(threshold_sampling(args.sketch_size))
        elif algo == 'CS_sketch':
            algorithms.append(CS_sketch(sketch_size=args.sketch_size))

    dataset_adaptor = load_dataset(args.dataset, args.dataset_name, config={})
    results = run_benchmark(algorithms, dataset_adaptor, args.num_runs, args.dataset, args.data_shape, args.sparsity, args.vector, args.type, args.precision, args.data_mean, args.data_deviation,args.sketch_size)

    for result in results:
        logging.info(f"Algorithm: {result['algorithm']}")
        logging.info(f"Metrics: {result['metrics']}")
        logging.info(f"Average Execution Time: {result['avg_execution_time']} seconds")
        logging.info("------------------------")

    save_results(results)

if __name__ == "__main__":
    main()