import argparse


def main():
    parser = argparse.ArgumentParser(description="Inner Product Estimation Benchmark")
    parser.add_argument('--algo', type=str, choices=['baseline', 'optimized'], required=True,action='append',
                        help="what algorithm(s) to use")
    parser.add_argument('--dataset', type=str, choices=['provided', 'generated'], required=True,
                        help="what dataset to use")
    parser.add_argument('--data_shape', type=int, nargs=4,
                        default=[1000, 500, 500, 1000],
                        help="scale of data, in format Matrix1(ROWS, COLS) Matrix2(ROWS, COLS),"
                             " Example: --data_scale 1000 500 500 1000")
    parser.add_argument('--data_mean', type=float,
                        default=10,
                        help="Mean value of generated data")
    parser.add_argument('--data_deviation', type=float,
                        default=1,
                        help="Deviation of generated data")
    parser.add_argument('--sparsity', type=float, required=True,
                        default=0,
                        help="sparsity parameter, how many elements in matrix or vector are zeros")
    parser.add_argument('--vector', action='store_true',
                        default=False,
                        help="using vector instead of matrix")
    parser.add_argument('--type', choices=['binary', 'normal'],
                        default="normal",
                        help="type of data, binary or normal")
    parser.add_argument('--precision', choices=['float', 'int'],
                        default='int',
                        help="precision type, float or int")
    parser.add_argument('--metrics', type=str, nargs='+', choices=['time', 'accuracy'],
                        default=['time', 'accuracy'],
                        help="list of metrics to evaluate")
    parser.add_argument('--num_runs', type=int,
                        default=5,
                        help="Number of runs to average the performance metrics.")

    args = parser.parse_args()
    algorithm = args.algo  # a list containing all algos to be tested
    dataset_type = args.dataset  # name of the specified dataset, provided or self-generated
    data_shape = args.data_shape  # a tuple (ROWS, COLS) of the scale of matrix
    data_mean = args.data_mean # mean value of generated data
    data_deviation = args.data_deviation # deviation of generated data
    sparsity = args.sparsity  # ratio of elements in matrix that is zero
    use_vector = args.vector  # use vector instead of matrix
    data_type = args.type  # using binary value or normal value
    precision = args.precision  # using float or int in matrix
    metrics = args.metrics  # a list of metrics that is used
    number_of_runs = args.num_runs  # number of runs

    #todo load dataset according to args
    if dataset_type == 'generated':
        pass
    else:
        #
        pass
    for run in range(0,number_of_runs):
        #todo: do each experiment
        pass
if __name__ == '__main__':
    main()
