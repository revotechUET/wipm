## WIPM Service

Service uses asynchronous queue to perform machine learning and it uses SocketIo to obtain non-time-dependent result. 

Service uses redis as the broker for the task queue and the place to store the returned results. 

This asynchronous queue is installed on the Celery framework

### Version dependency package

Python: 3.6.x

### Install dependency package

```bash
$ sudo apt update

$ sudo apt install redis python3 python3-tk python3-dev python3-pip libmysqlclient-dev 

$ mkdir -p files/{crp,classification,anfis,himpe} log

$ export LC_ALL=C

$ pip install -r requirements.txt

```

### Deploy service

```bash
$ pm2 start
```

### Deploy with docker

```bash
$ docker-compose up
```

### Notes

* If you encounter the error 'no display name and no $DISPLAY environment variable'

- Create ~/.config/matplotlib/matplotlibrc and add 'backend : Agg' to it

- Set the DISPLAY enviroment variable as 

```bash
$ export $DISPLAY=0:0
```

### Resources

* Celery - Distributed Task Queue [http://www.celeryproject.org/](http://www.celeryproject.org/)
* Flask - Microframework for web [http://flask.pocoo.org/](http://flask.pocoo.org/)
* Scikit-learn - machine learning library [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
* Keras - deep learning library base on tensorflow [https://keras.io/](https://keras.io/)
* scipy - for numerical integration, interpolation, optimization, linear algebra and statistics. (optional) [https://www.scipy.org/](https://www.scipy.org/)
* numpy - is the fundamental package for scientific computing with Python[https://www.numpy.org/](https://www.numpy.org/)
