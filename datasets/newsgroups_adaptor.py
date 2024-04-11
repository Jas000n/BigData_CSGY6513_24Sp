import os
import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets.dataset_adaptor import DatasetAdaptor

class NewsgroupsAdaptor(DatasetAdaptor):
    def load_data(self):
        # 加载20newsgroup数据集
        raw_data_path = os.path.join(self.config['data_path'], 'raw', 'newsgroups.pkl')
        if os.path.exists(raw_data_path):
            with open(raw_data_path, 'rb') as f:
                newsgroups_train, newsgroups_test = pickle.load(f)
        else:
            newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

            # 只使用前10000条文章
            newsgroups_train.data = newsgroups_train.data[:10000]
            newsgroups_train.target = newsgroups_train.target[:10000]

            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            with open(raw_data_path, 'wb') as f:
                pickle.dump((newsgroups_train, newsgroups_test), f)
        return newsgroups_train.data, newsgroups_train.target, newsgroups_test.data, newsgroups_test.target

    def preprocess_data(self, data):
        # 对文本数据进行预处理,如去除停用词,提取TF-IDF特征等
        processed_data_path = os.path.join(self.config['data_path'], 'processed', 'newsgroups.pkl')
        if os.path.exists(processed_data_path):
            with open(processed_data_path, 'rb') as f:
                x_train, y_train, x_test, y_test = pickle.load(f)
        else:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            x_train = vectorizer.fit_transform(data[0]).toarray()
            x_test = vectorizer.transform(data[2]).toarray()
            y_train, y_test = data[1], data[3]
            os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
            with open(processed_data_path, 'wb') as f:
                pickle.dump((x_train, y_train, x_test, y_test), f)
        return x_train, y_train, x_test, y_test