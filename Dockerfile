FROM ubuntu:18.04

WORKDIR /app

EXPOSE 5000 5555

ENV DISPLAY="0:0" LC_ALL="C" DEBIAN_FRONTEND="noninteractive"

RUN apt update -y && apt install -y curl redis python3 python3-dev python3-tk python3-pip libmysqlclient-dev

RUN curl -sL https://deb.nodesource.com/setup_11.x | bash - && apt install -y nodejs

RUN mkdir -p files/{crp,classification,anfis,himpe,regression} log

COPY requirements.txt .

RUN npm install -g pm2 && pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT ./docker-entrypoint.sh

CMD /bin/bash
