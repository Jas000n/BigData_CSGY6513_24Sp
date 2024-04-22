import time
from algorithms.algorithm_adaptor import algorithm_adaptor
from sklearn.random_projection import SparseRandomProjection

class TensorSketchAdaptor(algorithm_adaptor):
    def __init__(self, name, sketch_size):
        super().__init__(name)
        self.sketch_size = sketch_size
        self.sketch = SparseRandomProjection(n_components=sketch_size)

    def inner_product_estimate(self, matrix1, matrix2):
        start_time = time.time()

        matrix1 = algorithm_adaptor.adjust_array_dimension(matrix1)
        matrix2 = algorithm_adaptor.adjust_array_dimension(matrix2)

        sketch1 = self.sketch.fit_transform(matrix1)
        sketch2 = self.sketch.transform(matrix2)
        estimate = sketch1.multiply(sketch2).sum()
        end_time = time.time()
        execution_time = end_time - start_time
        return estimate, execution_time