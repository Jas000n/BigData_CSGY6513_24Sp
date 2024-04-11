class AlgorithmAdaptor:
    def __init__(self, config):
        self.config = config

    def fit(self, data):
        raise NotImplementedError

    def query(self, data, k):
        raise NotImplementedError

    def evaluate(self, data):
        raise NotImplementedError
