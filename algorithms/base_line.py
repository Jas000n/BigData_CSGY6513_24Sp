from algorithms.algorithm_adaptor import algorithm_adaptor


class base_line(algorithm_adaptor):
    def __init__(self, sketch_size):
        super().__init__("Base_Line")
        self.sketch_size = sketch_size

    def sketch(self, vector, sketch_size):

        return vector