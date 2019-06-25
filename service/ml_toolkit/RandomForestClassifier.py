# Random Forest Classification

# Importing the libraries
from sklearn.ensemble import RandomForestClassifier as Model

from .Classifier import *

class RandomForestClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=None,
                 val_size=0.2, feature_cols=None, label_col=-1,
                 feature_degree=1, include=None, preprocess=False):

        super().__init__(train_set, val_set, data_file, header, val_size,
                         feature_cols, label_col, feature_degree, True,
                         include, preprocess, True, False)

    def fit(self, num_trees=150, criterion='entropy', min_samples_split=5,
            min_impurity_decrease=0.0003, verbose=False):

        if verbose:
            print('Using Random Forest Classifier...')

        self.model = Model(n_estimators=num_trees,
                           criterion=criterion,
                           min_samples_split=min_samples_split,
                           min_impurity_decrease=min_impurity_decrease)

        self.model.fit(self.X_train, self.y_train)

        if verbose:
            print('\n--- Training result ---')
            accuracy, loss = self.evaluate_helper(self.X_train, self.y_train, 0, verbose)

        # self.his['acc'] = [accuracy]
        # self.his['loss'] = [loss]

        # Predicting the Test set results
        self.evaluate_test(verbose=verbose)

    def save(self, file_name=None):
        del self.X_train
        del self.y_train
        del self.X_val
        del self.y_val

        if file_name is None:
            file_name = 'rf_model_' + str(int(round(self.score*100,1)))
        joblib.dump(self, file_name)
