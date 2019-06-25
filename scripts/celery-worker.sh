#!/bin/bash
source env/bin/activate

celery -A service.worker worker --loglevel=ERROR
