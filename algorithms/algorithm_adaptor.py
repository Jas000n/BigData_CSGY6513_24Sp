import time


class algorithm_adaptor:
    def __init__(self, name):
        self.name = name

    def inner_product_estimate(self, matrix1, matrix2):
        start_time = time.time()

        # Placeholder for the actual inner product estimation logic
        # Here you can implement your algorithm to estimate the inner product
        # This is just a dummy implementation
        estimate = sum(matrix1[i] * matrix2[i] for i in range(len(matrix1)))

        end_time = time.time()
        execution_time = end_time - start_time

        return estimate, execution_time

    def sketch(self, vector, sketch_size):
        # return sketched vector with shape 1*sketch_size
        pass

