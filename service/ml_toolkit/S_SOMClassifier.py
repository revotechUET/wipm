from ml_toolkit.lvq_network import AdaptiveLVQ as Model
from ml_toolkit.Classifier import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from numpy.random import shuffle

class S_SOMClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=None,
                 val_size=0.2, feature_cols=None, label_col=-1,
                 feature_degree=1, include=None, preprocess=False):
        super().__init__(train_set, val_set, data_file, header, val_size,
                         feature_cols, label_col, feature_degree, False,
                         include, preprocess, True, False)

        self.feature_scaling = True
        self.sc = MinMaxScaler((-1, 1))
        if self.feature_scaling:
            self.X_train = self.sc.fit_transform(self.X_train)
            if len(self.X_val > 0):
                self.X_val = self.sc.transform(self.X_val)

    def fit(self, size=9, learning_rate=0.5, decay_rate=1, sigma=2,
            sigma_decay_rate=1, weights_init='pca', neighborhood='bubble',
            first_num_iteration=1000, first_epoch_size=None,
            second_num_iteration=1000, second_epoch_size=None,
            verbose=False):

        self.model = Model(n_rows=size, n_cols=size, learning_rate=learning_rate,
                           decay_rate=decay_rate, sigma=sigma,
                           sigma_decay_rate=sigma_decay_rate,
                           weights_init=weights_init,
                           neighborhood=neighborhood,
                           label_weight='exponential_distance')

        if not first_epoch_size:
            first_epoch_size = self.num_samples
        if not second_epoch_size:
            second_epoch_size = self.num_samples

        self.model.fit(self.X_train, self.y_train,
                       first_num_iteration, first_epoch_size,
                       second_num_iteration, second_epoch_size)

        # evaluate on test set

        pred = self.model.predict(X=self.X_val)

        self.cm = confusion_matrix(self.y_val, pred, labels=self.labels)

    def evaluate(self, X=None, y=None, data=None, data_file=None, header=None,
                 radius=0, verbose=False):

        if X is not None and y is not None:
            data = np.append(X, np.array(y).reshape(-1,1), axis=1)
        elif data is not None:
            pass
        elif data_file is not None:
            if verbose:
                print('\nEvaluating on ', data_file, '...', sep='')
                data = pd.read_csv(data_file, header=header).values
        else:
            raise RuntimeError('Missing data')

        data = filt_data(data[:, self.cols], self.include)
        X = data[:, :-1]
        y = data[:, -1]
        if self.preprocess:
            X = add_features(X)

        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)
        y = self.le.transform(y)

        pred = self.model.predict(X=X)
        accuracy = np.count_nonzero(y == pred) / len(y) * 100

        if verbose:
            print('Accuracy: ', accuracy)

        pred = smoothen(pred, radius)
        accuracy = np.count_nonzero(y == pred) / len(y) * 100
        if verbose and radius > 0:
            print('Accuracy after smoothening with radius =', radius, ': ', accuracy * 100)

        # self.cm = confusion_matrix(y, pred, labels=self.labels)

        return {'acc': accuracy, 'loss': None}

    def judge(self, X=None, y=None, data=None, data_file=None, header=None,
              radius=0, verbose=False, threshold=0.0):

        if X is not None and y is not None:
            data = np.append(X, np.array(y).reshape(-1,1), axis=1)
        elif data is not None:
            pass
        elif data_file is not None:
            if verbose:
                print('\nJudging on ', data_file, '...', sep='')
            data = pd.read_csv(data_file, header=header).values
        else:
            raise RuntimeError('Missing data')

        data = filt_data(data[:, self.cols], self.include)
        X = data[:, :-1]
        y = data[:, -1]
        if self.preprocess:
            X = add_features(X)

        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)
        y = self.le.transform(y)

        pred, prob = self.model.predict(X=X, confidence=True)

        pred = judge(pred, prob, threshold=threshold)

        pred = smoothen(pred, radius)

        # cm = confusion_matrix(y, pred, labels=self.labels+[-9999])

        return cm

    def probability(self, X=None, data_file=None, header=None):
        pass

    def predict(self, X=None, data_file=None, header=None, radius=0,
                threshold=0.0):

        if data_file is not None:
            data = pd.read_csv(data_file, header=header).values
            X = data[:, self.cols[:-1]]
        elif X is not None:
            X = np.array(X[:, self.cols[:-1]])
        else:
            raise RuntimeError('Missing data')

        if self.preprocess:
            X = add_features(X)
        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)

        pred, prob = self.model.predict(X=X, confidence=True)

        pred = self.le.inverse_transform(pred)
        pred = judge(pred, prob, threshold=threshold, null_type=None)
        pred = smoothen(pred, radius)

        return pred

    def get_result(self, X=None):
        pred = self.predict(X)

        return dict(target=pred.tolist())

    def plot(self, path=None):
        self.model.visualize(path)

    def save(self, file_name=None):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        if file_name is None:
            file_name = 'slvq_model'
        joblib.dump(self, file_name)
