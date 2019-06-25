"""
Implement function build instance model machine learning
"""

import os
import shutil
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from ml_toolkit.neural_network import MLPRegressor


def build_model(model_type, params):
    """Implement function build model
    Support LinearRegression, HuberRegression, LassoRegression,
    SupportVectorRegression, DecisionTreeRegressor, RandomForest, XGBoost and
    MultiPerceptron
    Apply normalization input data

    Parameters:
    -----------
    model_type: str, type of model, support LinearRegression, HuberRegressor,
          Lasso, DecisionTreeRegressor, SupportVectorRegressor and
          XGBRegressor
    params: dict, parameter responding each model
    Returns:
    --------
    estimator: instance of class model
    """
    support_type = ['LinearRegression', 'HuberRegressor', 'Lasso',
                    'DecisionTreeRegressor', 'RandomForestRegressor',
                    'SupportVectorMachine', 'XGBoost', 'MultiPerceptron']
    assert (model_type in support_type), 'Expected one of value {}'.format(','.join(support_type))

    steps = [('minmax-scaler', MinMaxScaler()),
             ('standard-scaler', StandardScaler()),
             ('polynomial', PolynomialFeatures(params.pop('degree', 1)))]

    # Choice model type
    if model_type == 'LinearRegression':
        steps.append(('model', LinearRegression(**params)))
    elif model_type == 'HuberRegressor':
        steps.append(('model', HuberRegressor(**params)))
    elif model_type == 'Lasso':
        steps.append(('model', Lasso(**params)))
    elif model_type == 'DecisionTreeRegressor':
        steps.append(('model', DecisionTreeRegressor(**params)))
    elif model_type == 'RandomForestRegressor':
        steps.append(('model', RandomForestRegressor(**params)))
    elif model_type == 'SupportVectorMachine':
        steps.append(('model', SVR(**params)))
    elif model_type == 'XGBoost':
        steps.append(('model', XGBRegressor(**params)))
    elif model_type == 'MultiPerceptron':
        steps.append(('model', MLPRegressor(**params)))

    estimator = Pipeline(steps)
    return estimator


def save_model(model, save_path=os.getcwd()):
    """Implement function save model to disk
    Parameters:
    -----------
    model: object
    save_path: str

    Returns:
    --------
    """
    if os.path.exists(save_path):
        if os.path.isfile(save_path):
            os.remove(save_path)
        else:
            shutil.rmtree(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as export_f:
        pickle.dump(model, export_f)
    return None


def load_model(model_path):
    """Implement function load model from disk
    Parameters:
    -----------
    model_path: str, path to model storage

    Returns:
    --------
    estimator: object
    """
    with open(model_path, 'rb') as model_f:
        estimator = pickle.load(model_f)
    return estimator

