# Decision Tree Classification

# Importing the libraries
from sklearn.tree import DecisionTreeClassifier as Model

from ml_toolkit.Classifier import *

class DecisionTreeClassifier(Classifier):

    def __init__(self, train_set=None, val_set=None, data_file=None, header=None,
                 val_size=0.2, feature_cols=None, label_col=-1,
                 feature_degree=1, include=None, preprocess=False):
        super().__init__(train_set, val_set, data_file, header, val_size,
                         feature_cols, label_col, feature_degree, True,
                         include, preprocess, True, False)

    def fit(self, criterion='entropy', min_samples_split=5, min_impurity_decrease=0.01,
            verbose=False):

        # Fitting Decision Tree Classification to the Training set
        if verbose:
            print('Using Decision Tree Classifier...')

        self.model = Model(criterion=criterion,
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
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        if file_name is None:
            file_name = 'dt_model_' + str(int(round(self.score*100,1)))
        joblib.dump(self, file_name)
