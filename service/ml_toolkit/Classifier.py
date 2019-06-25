# Importing the libraries
import os
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier

from .clf_helper import *
confusion_matrix = cm_with_percentage

class Classifier:

    def __init__(self, train_set, val_set, data_file, header, val_size,
                 feature_cols, label_col, feature_degree, feature_scaling,
                 include, preprocess, proportional, keep_order):

        if preprocess:
            keep_order = True

        if not feature_cols:
            if train_set is None:
                feature_cols = list(range(6))
            else:
                feature_cols = list(range(train_set.shape[1]-1))

        self.cols = feature_cols + [label_col]
        self.include = include
        self.preprocess = preprocess

        if data_file is not None:
            data = pd.read_csv(data_file, header=header).values
            data = filt_data(data[:, self.cols], self.include)
            if self.preprocess:
                data = pre_process(data)
            train_set, val_set = split_data(data, val_size, proportional, keep_order)
        elif train_set is not None:
            if val_set is None:
                train_set = filt_data(train_set[:, self.cols], self.include)
                if self.preprocess:
                    train_set = pre_process(train_set)
                train_set, val_set = split_data(train_set, val_size, proportional, keep_order)
            else:
                train_set = filt_data(train_set[:, self.cols], self.include)
                val_set = filt_data(val_set[:, self.cols], self.include)
                if self.preprocess:
                    train_set = pre_process(train_set)
                    val_set = pre_process(val_set)
        else:
            raise RuntimeError('Missing data')

        self.X_train = train_set[:, :-1]
        self.X_val = val_set[:, :-1]
        self.y_train = train_set[:, -1].astype('int')
        self.y_val = val_set[:, -1].astype('int')

        self.le = LabelEncoder()
        self.y_train = self.le.fit_transform(self.y_train)
        self.y_val = self.le.transform(self.y_val)

        self.num_labels = max(self.y_train) + 1
        self.labels = [i for i in range(self.num_labels)]
        self.labels_origin = self.le.inverse_transform(self.labels).tolist()

        self.poly = PolynomialFeatures(feature_degree, include_bias=False)
        self.X_train = self.poly.fit_transform(self.X_train)
        if len(self.X_val > 0):
            self.X_val = self.poly.transform(self.X_val)

        self.num_samples, self.num_features = self.X_train.shape

        self.feature_scaling = feature_scaling
        self.sc = StandardScaler()
        if feature_scaling:
            self.X_train = self.sc.fit_transform(self.X_train)
            if len(self.X_val > 0):
                self.X_val = self.sc.transform(self.X_val)

        self.his = {'acc': None, 'loss': None, 'val_acc': None, 'val_loss': None}
        self.model = None
        self.cm = None
        self.score = 0

    def evaluate_helper(self, X, y, radius, verbose):
        prob = self.model.predict_proba(X)

        try:
            loss = log_loss(y, prob)
        except:
            loss = -1

        pred = self.model.predict(X=X)
        accuracy = np.count_nonzero(y == pred) / len(y)

        if verbose:
            print('Accuracy: ', accuracy * 100, ' Loss: ', loss)

        pred = smoothen(pred, radius)
        accuracy = np.count_nonzero(y == pred) / len(y)
        if verbose and radius > 0:
            print('Accuracy after smoothening with radius =', radius, ': ', accuracy * 100)

        self.cm = confusion_matrix(y, pred, labels=self.labels)

        return accuracy, loss

    def evaluate_test(self, radius=0, verbose=False):

        if len(self.X_val) > 0:
            if verbose:
                print('\nEvaluating on test set...')
            X = self.X_val
            y = self.y_val

            accuracy, loss = self.evaluate_helper(X, y, radius, verbose)

            self.score = accuracy * 100

    def evaluate(self, X=None, y=None, data=None, data_file=None, header=None,
                 radius=0, verbose=False):

        if X is not None and y is not None:
            data = np.append(X, np.array(y).reshape(-1,1), axis=1)
        elif data is not None:
            pass
        elif data_file is not None:
            if verbose:
                print('\nEvaluating on ', data_file, '...', sep='')
            data = pd.read_csv(data_file, header=header).values
        else:
            raise RuntimeError('Missing data')

        data = filt_data(data[:, self.cols], self.include)
        X = data[:, :-1]
        y = data[:, -1]
        if self.preprocess:
            X = add_features(X)

        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)
        y = self.le.transform(y)

        accuracy, loss = self.evaluate_helper(X, y, radius, verbose)

        return {'acc': accuracy, 'loss': loss}

    def judge(self, X=None, y=None, data=None, data_file=None, header=None,
              radius=0, verbose=False, threshold=0.8):

        if X is not None and y is not None:
            data = np.append(X, np.array(y).reshape(-1,1), axis=1)
        elif data is not None:
            pass
        elif data_file is not None:
            if verbose:
                print('\nJudging on ', data_file, '...', sep='')
            data = pd.read_csv(data_file, header=header).values
        else:
            raise RuntimeError('Missing data')

        data = filt_data(data[:, self.cols], self.include)
        X = data[:, :-1]
        y = data[:, -1]
        if self.preprocess:
            X = add_features(X)

        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)
        y = self.le.transform(y)

        prob = self.model.predict_proba(X)
        pred = prob.argmax(axis=1)
        confidence = np.max(prob, axis=1)

        pred = judge(pred, prob, threshold=threshold)

        pred = smoothen(pred, radius)

        cm = confusion_matrix(y, pred, labels=self.labels+[-9999])

        return cm

    def probability(self, X=None, data_file=None, header=None):

        if data_file is not None:
            data = pd.read_csv(data_file, header=header).values
            X = data[:, self.cols[:-1]]
        elif X is not None:
            X = np.array(X[:, self.cols[:-1]])
        else:
            raise RuntimeError('Missing data')

        if self.preprocess:
            X = add_features(X)
        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)

        return self.model.predict_proba(X)

    def predict(self, X=None, data_file=None, header=None, radius=0,
                threshold=0.0):

        if data_file is not None:
            data = pd.read_csv(data_file, header=header).values
            X = data[:, self.cols[:-1]]
        elif X is not None:
            X = np.array(X[:, self.cols[:-1]])
        else:
            raise RuntimeError('Missing data')

        if self.preprocess:
            X = add_features(X)
        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)

        prob = self.model.predict_proba(X)
        pred = prob.argmax(axis=1)
        confidence = np.max(prob, axis=1)

        pred = self.le.inverse_transform(pred)
        pred = judge(pred, confidence, threshold=threshold, null_type=None)
        pred = smoothen(pred, radius)

        return pred

    def get_result(self, X=None, data_file=None, header=None, radius=0,
                   threshold=0.0):

        if data_file is not None:
            data = pd.read_csv(data_file, header=header).values
            X = data[:, self.cols[:-1]]
        elif X is not None:
            X = np.array(X[:, self.cols[:-1]])
        else:
            raise RuntimeError('Missing data')

        if self.preprocess:
            X = add_features(X)
        X = self.poly.transform(X)
        if self.feature_scaling:
            X = self.sc.transform(X)

        prob = self.model.predict_proba(X)
        pred = prob.argmax(axis=1)
        confidence = np.max(prob, axis=1)

        pred = self.le.inverse_transform(pred)
        pred = judge(pred, confidence, threshold=threshold, null_type=None)
        pred = smoothen(pred, radius)

        cum_prob = cumulate_prob(prob)

        return dict(target=pred.tolist(), prob=cum_prob.tolist())

    def get_cm_data_url(self, id):
        if self.cm is None:
            return None

        draw_confusion_matrix(self.cm, self.labels_origin + [''])
        img = id + '.png'
        plt.savefig(img)
        data_url = image_to_data_url(img)
        os.remove(img)

        return data_url

    @classmethod
    def load(Classifier, file_name):
        return joblib.load(file_name)
