class DatasetAdaptor:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        raise NotImplementedError

    def preprocess_data(self, data):
        raise NotImplementedError