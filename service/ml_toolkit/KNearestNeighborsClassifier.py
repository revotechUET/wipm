# K-Nearest Neighbors (K-NN)

# Importing the libraries
from sklearn.neighbors import KNeighborsClassifier as Model

from ml_toolkit.Classifier import *

class KNearestNeighborsClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=None,
                 val_size=0.2, feature_cols=None, label_col=-1,
                 feature_degree=1, include=None, preprocess=False):

        super().__init__(train_set, val_set, data_file, header, val_size,
                         feature_cols, label_col, feature_degree, True,
                         include, preprocess, True, False)

    def fit(self, num_neighbors=100, p=1, verbose=False):

        if verbose:
            print('Using K-Nearest Neighbors Classifier...')

        self.model = Model(n_neighbors=num_neighbors, metric='minkowski', p=p)

        self.model.fit(self.X_train, self.y_train)

        if verbose:
            print('\n--- Training result ---')
            accuracy, loss = self.evaluate_helper(self.X_train, self.y_train, 0, verbose)
        #
        # self.his['acc'] = [accuracy]
        # self.his['loss'] = [loss]

        self.evaluate_test(verbose=verbose)

    def save(self, file_name=None):
        del self.X_train
        del self.y_train
        del self.X_val
        del self.y_val

        if file_name is None:
            file_name = 'knn_model_' + str(int(round(self.score*100,1)))
        joblib.dump(self, file_name)
