"""Setting default params for all model machine learning
Define Factory class for get model
"""

from ml_toolkit.Classifier import Classifier
from ml_toolkit.DecisionTreeClassifier import DecisionTreeClassifier
from ml_toolkit.KNearestNeighborsClassifier import KNearestNeighborsClassifier
from ml_toolkit.LogisticRegressionClassifier import LogisticRegressionClassifier
from ml_toolkit.NeuralNetworkClassifier import NeuralNetworkClassifier
from ml_toolkit.RandomForestClassifier import RandomForestClassifier
from ml_toolkit.S_SOMClassifier import S_SOMClassifier
from ml_toolkit.HIMFAClassifier import HIMFAClassifier

CLASSIFIER = {
    'NeuralNetClassifier': NeuralNetworkClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'KNN': KNearestNeighborsClassifier,
    'LogisticRegression': LogisticRegressionClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'HIMFA': HIMFAClassifier,

    'S_SOM': S_SOMClassifier,
}

DEFAULT_PARAMS = {
    'LinearRegression': {
        'normalize': False,
        'fit_intercept':True,
        'degree': 2,
        'copy_X': True,
        'n_jobs': 1
    },
    'Lasso':{
        'alpha': 50,
        'degree': 2,
        'precompute': False,
        'max_iter': 1000,
        'tol': 1,
        'positive': False,
        'selection': 'random',
    },
    'RandomForestRegressor':{
        'n_estimators': 300,
        'criterion': 'mse',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': 'log2',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'bootstrap': True,
        'oob_score': False,
        'n_jobs': 4
    },
    'SupportVectorMachine':{
        'kernel': 'linear',
        'degree': 2,
        'gamma': 'auto',
        'coef0': 0.0,
        'tol': 0.001,
        'C': 1,
        'epsilon': 1e-5,
        'shrinking': True,
        'cache_size': 200,
        'max_iter': -1
    },
    'XGBoost': {
        'max_depth': 20,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'silent': True,
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'n_jobs': 4,
        'nthread': None,
        'gamma': 0.0,
        'min_child_weight': 1,
        'max_delta_step': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1,
        'base_score': 0.5,
        'random_state': None,
        'seed': 0,
        'missing': None
    },
    'MultiPerceptron':{
        'solver': 'lbfgs',
        'learning_rate_init': 0.001,
        'max_iter': 200,
        'activation': 'relu',
        'hidden_layer_sizes': (10,10,),
        'alpha': 1e-2,
        'tol': 0.0003,
        'random_state': 7,
        'strategy': 'best1bin',
        'recombination': None,
        'popsize': 5,
        'bound': (-10,10),
        'verbose':False
    },
    'DecisionTreeRegressor':{
        'criterion': 'mse',
        'splitter': 'best',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': None,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'presort': False
    },
    'HuberRegressor': {
        'epsilon': 1.35,
        'max_iter': 1000,
        'alpha': 0.0001,
        'degree': 3,
        'fit_intercept': True,
        'tol': 1e-05
    },

    'NeuralNetClassifier': {
        'hidden_layer_sizes': [10, 20, 10,],
        'activation': 'elu',
        'algorithm': 'backprop',
        'batch_size': None,
        'learning_rate': 0.001,
        'num_epochs': 10000,
        'optimizer': 'nadam',
        'warm_up': False,
        'boosting_ops': 0,
        'sigma': 0.01,
        'population': 50
    },
    'DecisionTreeClassifier': {
        'criterion': 'entropy',
        'min_samples_split': 5,
        'min_impurity_decrease': 0.01
    },
    'KNN': {
        'num_neighbors': 100,
        'p': 1
    },
    'LogisticRegression': {
        'c': 20,
        'max_iter': 10000,
        'solver': 'liblinear'
    },
    'RandomForestClassifier': {
        'num_trees': 150,
        'criterion': 'entropy',
        'min_samples_split': 5,
        'min_impurity_decrease': 0.0003
    },
    'HIMFA': {},

    'S_SOM': {
        'size': 9,
        'learning_rate': 0.5,
        'decay_rate': 1,
        'sigma': 2,
        'sigma_decay_rate': 1,
        'weights_init': 'pca',
        'neighborhood': 'bubble',
        'first_num_iteration': 1000,
        'first_epoch_size': None,
        'second_num_iteration': 1000,
        'second_epoch_size': None
    }
}

def params_filter(model_type, params=None):
    """Implement function filter params with model_type
    Parameters:
    -----------
    model_type: str
    params: dict

    Returns:
    --------
    default_params: dict, normalize params
    """
    default_params = DEFAULT_PARAMS[model_type]
    keys = list(params.keys())
    if params:
        for key in keys:
            if key not in default_params:
                del params[key]
        default_params.update(params)
    return default_params

class ModelFactory(object):
    @staticmethod
    def get_classifier(train_set=None, model_type=None):
        model = Classifier
        if model_type != None:
            model = CLASSIFIER[model_type](train_set=train_set)
        return model
