from datasets.dataset_adaptor import DatasetAdaptor
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

class NewsGroup20Adaptor(DatasetAdaptor):
    def __init__(self, config):
        super().__init__(config)

    def load_data(self):
        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')

        train_data = resample(newsgroups_train.data, n_samples=100, random_state=42)
        test_data = resample(newsgroups_test.data, n_samples=100, random_state=42)

        return train_data, test_data

    def preprocess_data(self, data):
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(data[0])
        X_test = vectorizer.transform(data[1])
        return X_train, X_test