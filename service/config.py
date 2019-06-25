"""Contain all config setting for module
machine learning
"""

from os import environ as env

# Load config object
config_object = dict(server={}, worker={}, path={}, monitor={})

config_object['worker']['broker_url'] = env.get('BROKER_URL', 'redis://127.0.0.1:6379/0')
config_object['worker']['result_backend'] = env.get('CELERY_RESULT_BACKEND', 'mongodb://127.0.0.1:27017/celery')
config_object['worker']['result_serializer'] = env.get('CELERY_RESULT_SERIALIZER', 'json')

config_object['server']['host'] = env.get('WIPM_HOST', '127.0.0.1')
config_object['server']['debug'] = env.get('WIPM_DEBUG', 1)
config_object['monitor']['tasks_columns'] = ['name', 'uuid', 'state', 'args',\
        'kwargs', 'result', 'received', 'started', 'runtime', 'worker',\
        'retries', 'revoked', 'exception', 'expires', 'eta']

config_object['path']['reg'] = env.get('REG_PATH', './files/regression')
config_object['path']['clf'] = env.get('CLF_PATH', './files/classification')
config_object['path']['crp'] = env.get('CRP_PATH', './files/crp')
config_object['path']['anfis'] = env.get('ANFIS_PATH', './files/anfis')
config_object['path']['himpe'] = env.get('HIMPE_PATH', './files/himpe')
