"""Receive task from client, then push mosquitto broker
"""

import os
import sys
import logging
from shutil import rmtree
from functools import wraps

from flask import request, Flask, jsonify
from flask_socketio import SocketIO, emit
import numpy as np

from service.worker import task
from service.config import config_object
from service.verify import ParametersException

logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
server_logger = logging.getLogger('server')
server_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.DEBUG)
log_handler.setFormatter(formatter)
server_logger.addHandler(log_handler)

server = Flask(__name__)
socketio = SocketIO(server)


@server.after_request
def apply_caching(response):
    """Solver issue CORS
    """
    response.headers["Access-Control-Allow-Origin"] = '*'
    response.headers["Access-Control-Allow-Methods"] = 'POST'
    response.headers["Access-Control-Allow-Headers"] = 'Content-Type'
    return response

@server.route('/', methods=['GET'])
def index():
    return jsonify(message='server API', status=200), 200

@server.route('/', methods=['POST'])
def request_task(*args, **kwargs):
    try:
        detail = kwargs.get('detail', {})
        payload = request.get_json()
        result = task.delay(payload)
        server_logger.debug('Received task with id %s' % result.id)
    except Exception as err:
        return jsonify(message=str(err), status=400), 400
    else:
        return jsonify(message="SUCCESS", status=200, task_id=result.id, detail=detail), 200
    finally:
        pass

@server.route('/model/delete', methods=['POST'])
def delete_model():
    try:
        MODEL_CURVES_CLF = config_object['path']['clf']
        MODEL_CURVES_REG = config_object['path']['reg']
        req = request.get_json()
        model_id = req.get('model_id', None)
        task_id = req.get('task_id', None)

        if model_id is None:
            raise ValueError('model_id is missing')
        else:
            try:
                model_path = os.path.join(MODEL_CURVES_REG, model_id)+'.pickle'
                if os.path.isfile(model_path):
                    os.remove(model_path)
                else:
                    model_path = os.path.join(MODEL_CURVES_CLF, model_id)
                    if os.path.isdir(model_path):
                        rmtree(model_path)
                    else:
                        model_path += '.pickle'
                        os.remove(model_path)
            except FileNotFoundError:
                raise ValueError('Model %s not exist' % model_id)
            else:
                # delete result train for neural network
                if task_id is not None:
                    res = task.AsyncResult(task_id=task_id)
                    res.forget()
    except Exception as error:
        return jsonify(message=str(error), status=200), 200
    else:
        return jsonify(message='SUCCESS', status=200), 200
    finally:
        pass

@socketio.on('connect')
def handler_connect():
    server_logger.debug('User connected')

@socketio.on('disconnect')
def handler_disconnect():
    server_logger.debug('User disconnect')

@socketio.on_error
def error_handler(error):
    server_logger.debug(str(error))

@socketio.on('post')
def handler_result_task(json):
    task_id = json['task_id']
    del_result = json.get('del_result', False)
    response = dict(message='', status=200, result={}, traceback='')
    res = task.AsyncResult(task_id=task_id)
    if res.status == 'PENDING':
        response['message'] = 'PENDING'
    elif res.state == 'SUCCESS':
        response['message'] = 'SUCCESS'
        response['result'] = res.get()
        if del_result:
            res.forget()
    else:
        if isinstance(res.result, ParametersException):
            response['message'] = str(res.result)
        else:
            response['message'] = 'FAILURE'
        if bool(config_object['server']['debug']):
            response['traceback'] = res.traceback
        response['status'] = 400
    emit(task_id, response)
    server_logger.debug('Emit result of task %s, size %d, status %d' % \
                         (task_id, sys.getsizeof(response), response['status']))

