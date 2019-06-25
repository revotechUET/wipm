#!/bin/bash

source env/bin/activate

flower -A service.worker --host=0.0.0.0 --port=5555 --basic_auth=wipm-admin:123456
