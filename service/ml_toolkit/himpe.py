
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss,\
        accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skcmeans.algorithms import Probabilistic
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPRegressor


def input_fn(ftrain, fval):
    """Specify process load data for this paper

    Parameters
    ----------
    ftrain: string, path to file csv dataset for trainning

    fval: string, path to file csv dataset for validation

    Returns
    -------
    df_train: pandas DataFrame, table training set with header and no index

    df_val: pandas DataFrame, table training set with header and no index

    Notes
    -----
    File must be .csv format with first line is header and no index
    """
    df_train = pd.read_csv(ftrain, header=0, index_col=None)
    if fval is not None:
        df_val = pd.read_csv(fval, header=0, index_col=None)
    else:
        df_val = None

    return df_train, df_val


def clustering(dataset, n_clusters=6, by='target'):
    """Clustering process perform cluster dataset by data or target

    Parameters
    ----------
    dataset: pandas DataFrame, default last column is target

    n_clusters: int, number of clusters

    by: string, `data` or `target`, target for clustering process, default `target`

    Returns
    -------
    dataset_with_labels: pandas DataFrame, dataset assigned label, columns labels is last
    """
    distances = []
    data = dataset.values[:, :-1]
    target = dataset.values[:, -1].reshape(-1, 1)
    cluster = Probabilistic(n_clusters=n_clusters, random_state=1)

    if by not in ['data', 'target']:
        print('No support')
        return None
    if by == 'target':
        cluster.fit(target)
        distances = cluster.distances(target)
    else:
        cluster.fit(data)
        distances = cluster.distances(data)

    labels = np.argmin(distances, axis=1)
    labels = pd.DataFrame(data=labels, columns=['LABEL'])
    dataset_with_labels = pd.concat([dataset, labels], axis=1)

    return dataset_with_labels


def gen_classifier(dataset, disp=True):
    """Perform setting and generate a classifier from DataFrame with labels

    Parameters
    ----------
    dataset: pandas DataFrame, default last column is labels

    disp: bool, display report if True

    Returns
    -------
    classifier: object of clf, trained with data
    """
    data = dataset.values[:, :-2]
    labels = dataset.values[:, -1]
    steps = [('std', StandardScaler()),
             ('estimator', XGBClassifier(n_estimators=300, max_depth=30))]
    classifier = Pipeline(steps)
    classifier.fit(data, labels)

    if disp:
        loss = 0
        accuracy = 0
        labels_pred = classifier.predict(data)
        # loss = log_loss(y_pred=labels_pred, y_true=labels)
        accuracy = accuracy_score(y_pred=labels_pred, y_true=labels)
        cm = confusion_matrix(y_pred=labels_pred, y_true=labels)
        print('Classifier report')
        print('-----------------')
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')
        plt.imshow(X=cm, cmap=plt.cm.Blues, interpolation='nearest')
        plt.title('Confusion matrix for classifier')
        plt.colorbar()
        plt.show()

    return classifier


def gen_estimators(dataset, disp=True):
    """Perform generate model regression for PERM

    Parameters
    ----------
    dataset: pandas DataFrame, default last colum is labels

    disp: bool, display report if True

    Returns
    -------
    estimators: list of regressor
    """
    n_models = len(set(dataset.values[:, -1]))
    estimators = []
    for i in range(n_models):
        steps = [('std', MinMaxScaler()),
                 ('estimator', MLPRegressor(
                     solver='adam',
                     max_iter=1000,
                     hidden_layer_sizes=(10, 20, 30, 10, )))]
        estimators.append(Pipeline(steps))
    groups = dataset.groupby('LABEL')

    for group in groups:
        index = group[0]
        data = group[1].values[:, :-2]
        target = group[1].values[:, -2]
        estimators[index].fit(data, target)

        if disp:
            target_pred = estimators[index].predict(data)
            mae = mean_absolute_error(y_true=target, y_pred=target_pred)
            mse = mean_squared_error(y_true=target, y_pred=target_pred)
            error = abs(target - target_pred)
            print(f'Estimator {index} report')
            print('-------------------')
            print(f'MSE: {mse}')
            print(f'MAE: {mae}')
            plt.hist(error, bins=50)
            plt.title('Distributed error')
            plt.show()

    return estimators


def save_models(classifier, estimators, fmodels=None):
    """Perform combine and dump pickle to file with name is fmodels

    Parameters
    ----------
    classifier: object, it trained

    estimators: list of object, it trained

    fmodels: string, if fmodel is not None, it is name of dump file

    Returns
    -------
    models: dict, keys: classifier and estimators
    """
    models = dict(classifier=classifier, estimators=estimators)
    if fmodels is not None:
        with open(fmodels, 'wb') as dump_file:
            pickle.dump(models, dump_file)
    return models


def predict(models, data):
    """Perform predict data using models and data

    Parameters
    ----------
    models: dict, models combined
    data: ndarray (n_samples, 6), a numpy array has 6 features

    Returns
    -------
    perm: ndarray (n_samples,), a 1d numpy array is perm
    """
    labels = models['classifier'].predict(data).astype(int)
    y_pred = []
    for i in range(len(labels)):
        index = labels[i]
        x = data[i]
        y = models['estimators'][index].predict([x])
        y_pred.append(y[0])

    y_pred = np.array(y_pred).reshape(-1)

    return y_pred


def create_models(ftrain='train.csv', fval='val.csv', n_clusters=8, fmodels='himpe'):
    """Perform create models

    Parameters
    ----------
    ftrain: string, path to csv file training dataset
    fval: string, path to csv file training dataset
    n_clusters: int, numbe of cluster
    fmodels: string, path to dump file models

    Returns
    -------
    models: dict
    """
    df_train, df_val = input_fn(ftrain, fval)
    df_with_label = clustering(dataset=df_train, by='data', n_clusters=n_clusters)
    classifer = gen_classifier(dataset=df_with_label, disp=True)
    estimators = gen_estimators(dataset=df_with_label, disp=True)
    models = save_models(classifier=classifer, estimators=estimators, fmodels=fmodels)
    if df_val is not None:
        evaluate(df_val, models, False)

    return models


def evaluate(df_val, models, disp=True):
    """Perform evaluate models

    Parameters
    ----------
    df_val: pandas DataFrame, with PERM
    models: dict, models himpe
    disp: bool, display plot if True

    Returns
    -------
    None
    """
    x_val = df_val.values[:, :-1]
    y_val = df_val.values[:, -1]
    y_pred = predict(models=models, data=x_val)
    x = list(range(len(y_val)))
    error = abs(y_val-y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    print('Models report')
    print('-------------')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')

    if disp:
        plt.subplots(121)
        plt.hist(error, bins=100)
        plt.xlabel('Error')
        plt.ylabel('n sample')
        plt.title('Report error')
        plt.subplots(122)
        plt.plot(x, y_val, label='Actual')
        plt.plot(x, y_pred, label='Predict')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    import sys

    if sys.argv[1] == 'create':
        create_models()
    elif sys.argv[1] == 'load':
        with open('himpe', 'rb') as f:
            models = pickle.load(f)
        df_val = pd.read_csv('val.csv', header=0, index_col=None)
        y_pred = predict(models, df_val.values[:, :-1])
        mae = mean_absolute_error(df_val.values[:, -1], y_pred)
        print(f'MAE: {mae}')
