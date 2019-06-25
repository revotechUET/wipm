#!/bin/sh

redis-server --daemonize yes

pm2 start

pm2 logs -f ml-server
