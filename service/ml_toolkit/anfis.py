import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skcmeans.algorithms import Probabilistic

headers = ['GR', 'NPHI', 'RHOB', 'DT', 'VCL', 'PHIE', 'PERM_CORE']

class Anfis(object):
    def __init__(self, n_clusters=[4,6], hidden_layer_sizes=(10, 10, 10, ), \
                 activation='relu', solver='adam', alpha=0.0001, \
                 batch_size='auto', learning_rate='constant', \
                 learning_rate_init=0.001, power_t=0.5, max_iter=200, \
                 shuffle=True, random_state=None, tol=0.0001, verbose=True, \
                 warm_start=False, momentum=0.9, nesterovs_momentum=True, \
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, \
                 beta_2=0.999, epsilon=1e-08, is_log_target=False, is_scaler_data=True):
        self.n_clusters = n_clusters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.is_log_target = is_log_target
        self.is_scaler_data = is_scaler_data
        self.model_1 = DecisionTreeRegressor(max_depth=6)
        self.model_2 = MLPRegressor(self.hidden_layer_sizes, self.activation, \
                                    self.solver, self.alpha, max_iter=1000)

    def fit(self, X, y):
        # create numpy array for input
        if isinstance(X, np.ndarray):
            X = np.array(X)
        if isinstance(y, np.ndarray):
            y = np.array(y)
        y = y.reshape(-1, 1)

        self.model_1.fit(X, y)

        # create instance
        if self.is_scaler_data:
            self._scaler = MinMaxScaler()
        if self.is_log_target:
            self._log = np.log10
        self._clusters = [ Probabilistic(n_clusters=i) for i in self.n_clusters ]

        # cluster, combine and scale data
        self._clusters[0].fit(X)
        self._clusters[1].fit(y)
        _f1 = np.argmin(self._clusters[0].distances(X), axis=1).reshape((-1, 1))
        _f2 = np.argmin(self._clusters[1].distances(y), axis=1).reshape((-1, 1))
        X = np.concatenate([X, _f1, _f2], axis=1)

        if self.is_scaler_data:
            X = self._scaler.fit_transform(X)
        if self.is_log_target:
            y = self._log(y)


        self.model_2.fit(X, y)

    def predict(self, X):
        y_pseudo = self.model_1.predict(X).reshape(-1, 1)
        _f1 = np.argmin(self._clusters[0].distances(X), axis=1).reshape((-1, 1))
        _f2 = np.argmin(self._clusters[1].distances(y_pseudo), axis=1).reshape((-1, 1))
        X = np.concatenate([X, _f1, _f2], axis=1)
        if self.is_scaler_data:
            X = self._scaler.transform(X)
        y = self.model_2.predict(X)
        if self.is_log_target:
            y = y**10
        return y.reshape(-1)

    def evaluate(self, X, y):
        pass


