"""Contain all function machine learning
"""

import os
import pickle
from shutil import rmtree
from random import randint as rint
from functools import wraps

from celery import Celery
import numpy as np
import pandas as pd
import matplotlib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
import tensorflow as tf

os.environ['DISPLAY'] = '0:0'
matplotlib.use('Agg')

from ml_toolkit import regression
from ml_toolkit import ml_toolkit_helper
from ml_toolkit.clf_util import *
from ml_toolkit.himpe import predict as himpe_predict
from ml_toolkit.crp import get_data_from_json
from ml_toolkit.crp import train as crp_train
from ml_toolkit.crp import predict as crp_predict

from service.config import config_object
import service.verify as verify


app = Celery('ml_task')
app.conf.update(config_object['worker'])

@verify.lower_case_params
def _regression_train(payload):
    """Implement task training model regression
    Parameters:
    -----------
    payload: dict
        - model_id: string, identify of model, using setup dir name when save
                    model
        - model_type: string, type of model, support LinearRegression,
                    LassoRegression, HuberRegression, DecisionTreeRegressor,
                    RandomForestRegressor, SupportVectorRegressor, XGBoost and
                    MultiPerceptron
        - train: dict contain numpy array dataset
            - data: ndarray, shape (n_features, n_samples)
            - target: ndarray, shape (n_samples, )
        - params: dict of parameters

    Returns:
    --------
    result: dict
        - train_loss: list if type is MultiPerceptron else double
        - train_error: list if type is MultiPerceptron else double
        - val_loss: list if type is MultiPerceptron else double
        - val_error: list if type is MultiPerceptron else double
    """
    result = dict()
    data = np.array(payload['train']['data']).T
    target = np.array(payload['train']['target'])
    params = ml_toolkit_helper.params_filter(payload['model_type'], payload['params'])
    save_path = os.path.join(
            config_object['path']['reg'],
            payload['model_id']+'.pickle')


    model = regression.build_model(model_type=payload['model_type'],
                                   params=params)

    if payload['model_type'] == 'MultiPerceptron':
        model.fit(data, target)
        loss = model.named_steps['model'].lpath['train']
        val_loss = model.named_steps['model'].lpath['val']
        result['loss'] = loss
        result['val_loss'] = val_loss
    else:
        x_train, x_val, y_train, y_val = train_test_split(data, target,
                                                          test_size=0.1)
        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_val = model.predict(x_val)

        result['train_loss'] = metrics.mean_squared_error(y_train, y_pred_train)
        result['train_error'] = metrics.mean_absolute_error(y_train, y_pred_train)
        result['val_loss'] = metrics.mean_squared_error(y_val, y_pred_val)
        result['val_error'] = metrics.mean_absolute_error(y_val, y_pred_val)
    regression.save_model(model, save_path=save_path)
    return result

def _regression_predict(payload):
    """Implement task predict target from model
    Parameters:
    -----------
    payload: dict
        - model_id: str
        - data: list of list, shape (n_features, n_samples)

    Returns:
    --------
    result: dict
        - target: list, shape (n_samples)
    """
    result = dict(target=[])
    data = np.array(payload['data']).T
    model_path = os.path.join(
            config_object['path']['reg'],
            payload['model_id']+'.pickle')

    model = regression.load_model(model_path)
    y_pred = model.predict(data)

    result['target'] = y_pred.reshape(-1).tolist()
    return result

@verify.knn
@verify.himfa
@verify.lower_case_params
def _classification_train(payload):
    MODEL_CURVES_CLF = config_object['path']['clf']
    result = dict()
    data = np.array(payload['train']['data']).T
    target = np.array(payload['train']['target']).reshape(-1, 1)
    train_set = np.concatenate((data, target), axis=1)

    model = ml_toolkit_helper.ModelFactory.get_classifier(model_type=payload['model_type'], \
                                                   train_set=train_set)
    params = ml_toolkit_helper.DEFAULT_PARAMS[payload['model_type']].copy()
    params.update(payload.get('params', {}))
    model_id = payload['model_id']
    filename = os.path.join(MODEL_CURVES_CLF, model_id)
    if os.path.exists(filename):
        if os.path.isdir(filename):
            rmtree(filename)
        else:
            os.remove(filename)

    if payload['model_type'] == 'NeuralNetClassifier':
        if params['batch_size'] == 0:
            params['batch_size'] = None
        model.structure(params.pop('hidden_layer_sizes'), params.pop('activation'))
    elif payload['model_type'] == 'S_SOM':
        if params['first_epoch_size'] == 0:
            params['first_epoch_size'] = None
        if params['second_epoch_size'] == 0:
            params['second_epoch_size'] = None
    model.fit(**params)
    model.save(filename)
    result = model.his.copy()
    result['cm'] = model.get_cm_data_url(model_id)

    return result

def _classification_predict(payload):
    MODEL_CURVES_CLF = config_object['path']['clf']
    result = dict()
    data = np.array(payload['data']).T
    filename = os.path.join(MODEL_CURVES_CLF, payload['model_id'])
    try:
        model = load(filename)
    except FileNotFoundError:
        raise RuntimeError('The model must be trained before it is used')
    else:
        result = model.get_result(data)

    return result


def _crp_train(payload):
    result = dict()
    training_well_ts, dim, tau, epsilon, lambd, \
        percent, curve_number, facies_class_number, \
        model_id = get_data_from_json(payload)
    if facies_class_number not in training_well_ts[:, -1].reshape(-1):
        result = dict(message='Facies class not exist', status=400)

    filename = os.path.join(config_object['path']['crp'], payload['model_id'])
    crp_train(training_well_ts, dim, tau, epsilon,
          lambd, percent, curve_number, facies_class_number, filename)

    return result


def _crp_predict(payload):
    result = dict()
    testset = payload['input']
    label_enc = LabelEncoder()
    well = label_enc.fit_transform(
        np.array(testset['well']).reshape(-1, 1)).reshape(-1, 1)
    testset['data'] = np.array(testset['data']).T
    filename = os.path.join(config_object['path']['crp'], payload['model_id'])
    testing_well_ts = np.concatenate((well, testset['data']), axis=1)
    predict_vector = crp_predict(testing_well_ts, filename)
    result = dict(target=predict_vector)

    return result


def _himpe_predict(payload):
    result = dict()
    data = payload['data']
    data = np.array(data).T
    filename = os.path.join(config_object['path']['himpe'], 'himpe')
    with open(filename, 'rb') as dump_file:
        models = pickle.load(dump_file)
    y_pred = models.predict(data)
    result = dict(target=y_pred.tolist())

    return result

def _perm_predict(payload):
    result = dict()
    data = np.array(payload['data']).T
    filename = os.path.join(config_object['path']['anfis'], 'anfis')
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(data)
    result = dict(target=y_pred.tolist())

    return result

@app.task()
def task(payload):
    result = dict()
    type, req = payload['type'], payload['req']
    if type == 'train_regression':
        result = _regression_train(req)
    elif type == 'predict_regression':
        result = _regression_predict(req)
    elif type == 'train_classification':
        result = _classification_train(req)
    elif type == 'predict_classification':
        result = _classification_predict(req)
    elif type == 'train_crp':
        result = _crp_train(req)
    elif type == 'predict_crp':
        result = _crp_predict(req)
    elif type == 'predict_himpe':
        result = _himpe_predict(req)
    elif type == 'predict_perm':
        result = _perm_predict(req)

    return result
