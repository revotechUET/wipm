from sklearn.preprocessing import MinMaxScaler
from skcmeans import algorithms
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import numpy

SCORING = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'sle': mean_squared_log_error,
    'rmse': lambda y_true, y_pred: mean_squared_error(y_true=y_true, y_pred=y_pred) ** 0.5,
    'mdae': median_absolute_error,
    'r2': r2_score
}


class CMean(object):
    def __init__(self, n_clusters=2):
        self.cluster = algorithms.Probabilistic(
            n_clusters=n_clusters, n_init=20)

    def fit(self, x):
        self.cluster.fit(x)

    def clusters(self, x):
        dis = self.cluster.distances(x)
        labels = []
        for d in dis:
            labels.append(d.argmin())
        return numpy.array(labels)


def get_batch(batch_size, n_sample):
    n_batches = int(n_sample / batch_size)
    batches = []
    for idx in range(n_batches):
        batches.append([idx * batch_size, (idx + 1) * batch_size])
    if n_batches * batch_size < n_sample:
        batches.append([n_batches * batch_size, n_sample])
    return batches


def scaler(data, feature_range=(0, 1)):
    instance = MinMaxScaler(feature_range=feature_range)
    data = instance.fit_transform(data)
    return data
