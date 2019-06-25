"""Verify user config
"""

from functools import wraps
import numpy as np


class ParametersException(Exception):
    pass

def knn(func):
    @wraps(func)
    def wrap_func(*args, **kwargs):
        payload = args[0]
        if payload['model_type'] == 'KNN':
            num_samples = int(len(np.array(payload['train']['data']).T) * 0.8)
            num_neighbors = payload['params'].get('num_neighbors')
            if num_neighbors and num_samples < num_neighbors:
                raise ParametersException(
                        f'Number of training samples (receiced {num_samples}) \
                        must be >= number of neighbors (receiced {num_neighbors}). \
                        It is recommended to have number of training samples be \
                        much higher than number of neighbors')
        result = func(*args, **kwargs)
        return result
    return wrap_func

def himfa(func):
    @wraps(func)
    def wrap_func(*args, **kwargs):
        payload = args[0]
        unique_target = set(payload['train']['target'])
        if payload['model_type'] == 'HIMFA':
            facies_models = payload['params']['facies_models']
            labels = []
            for model in facies_models:
                if len(model['group']) == 0:
                    raise ParametersException('Group cannot be empty')
                labels += model['group']
            biggest = max(labels)
            smallest = min(labels)
            if smallest < 0 or biggest > 10:
                raise ParametersException('label must be between 0 and 10 (inclusive)')

            for label in labels:
                if label not in unique_target:
                    raise ParametersException(f'Label {label} is not in target')
        result = func(*args, **kwargs)
        return result
    return wrap_func

def lower_case_params(func):
    @wraps(func)
    def wrap_func(*args, **kwargs):
        payload = args[0]
        mod_params = {}

        for key, value in payload['params'].items():
            if isinstance(key, str): key = key.lower()
            if isinstance(value, str): value = value.lower()
            mod_params[key] = value
        args = list(args)
        payload['params'] = mod_params
        args[0] = payload
        args = tuple(args)
        result = func(*args, **kwargs)
        return result
    return wrap_func
