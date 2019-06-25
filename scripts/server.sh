#!/bin/bash

source env/bin/activate

gunicorn --worker-class eventlet --workers 1 --bind 0.0.0.0:5000 wsgi:app
