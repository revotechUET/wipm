import numpy as np
import pandas as pd
import seaborn as sns
from random import sample
import matplotlib.pyplot as plt
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def generate_features(data, radius, features='mean-std'):

    n, m = data.shape
    interval = radius * 2 + 1

    if features == 'mean-std':
        new_features = np.zeros((n, m*2))
        for i in range(0, n):
            if i < radius:
                tmp = data[0:interval]
            elif i >= n-radius:
                tmp = data[n-interval:]
            else:
                tmp = data[i-radius : i+radius+1]
            new_features[i] = np.concatenate((np.mean(tmp, axis=0), np.std(tmp, axis=0)))

    elif features == 'diff':
        new_features = np.zeros((n, m))
        for i in range(1, n):
            new_features[i] = data[i] - data[i-1]

    return new_features

def add_features(data, trim=False, radius=3):
    n, m = data.shape

    res = np.concatenate((data,
                          generate_features(data, radius, features='mean-std'),
                          # generate_features(data, radius, features='diff')
                          ),
                         axis=1)
    if trim:
        return res[radius:-radius]

    return res

def filt_data(data, set, exclude=False):
    if not set:
        return data
    return data[np.where(np.isin(data[:,-1], set, invert=exclude))]

def group_data(data, groups):
    res = data.copy()
    for i in range(len(data)):
        for j in range(len(groups)):
            if res[i,-1] in groups[j]:
                res[i,-1] = j
                break

    return res

def mode(a):
    if isinstance(a, np.ndarray):
        a = a.tolist()

    return max(a, key=a.count)

def smoothen(y, radius):
    if radius <= 0:
        return y

    n = len(y)
    interval = radius * 2 + 1
    if n < interval - 1:
        return y

    for i in range(n):
        if i < radius:
            y[i] = mode(y[:interval])
        elif i >= n-radius:
            y[i] = mode(y[n-interval:])
        elif (y[i] != y[i-1]) or (y[i] != y[i+1]):
            y[i] = mode(y[i-radius : i+radius+1])

    return y

def pre_process(data, clustering=False, trim=False, radius=3):
    # return data

    n = len(data)
    interval = 2 * radius + 1
    res = None

    if clustering:
        start = 0
        i = 1
        while i <= n:
            if i == n or data[start, -1] != data[i, -1]:
                h = i - start
                if h > interval - 1:
                    if trim:
                        end = i - interval + 1
                    else:
                        end = i
                    tmp = np.concatenate((add_features(data[start:i, :-1], trim, radius),
                                          data[start:end, -1].reshape(-1, 1)),
                                         axis=1)
                    if res is not None:
                        res = np.append(res, tmp, axis=0)
                    else:
                        res = tmp

                start = i
            i += 1
    else:
        res = np.concatenate((add_features(data[:, :-1], trim=False, radius=radius),
                              data[:, -1].reshape(-1, 1)),
                             axis=1)

    if res is None or (10*len(res) < n):
        return data

    return res

def split_data(data, val_size=0.2, proportional=False, keep_order=False):

    if val_size == 0:
        return data, data[:0]

    if proportional:
        y = data[:,-1]
        sort_idx = np.argsort(y)
        val, start_idx= np.unique(y[sort_idx], return_index=True)
        indices = np.split(sort_idx, start_idx[1:])
        val_indices = []

        for idx_list in indices:
            n = len(idx_list)
            val_indices += idx_list[(sample(range(n), round(n*val_size)))].tolist()

        train_set = np.delete(data, val_indices, axis=0)
        if keep_order:
            val_indices = sorted(val_indices)
        else:
            shuffle(val_indices)
            shuffle(train_set)

        return train_set, data[val_indices]

    else:
        if keep_order:
            n = len(data)
            val_indices = (sample(range(n), round(n*val_size)))
            val_indices = sorted(val_indices)
            return np.delete(data, val_indices, axis=0), data[val_indices]

        return train_test_split(data, test_size=val_size)

def judge(predict, confidence, threshold, null_type=-9999):
    judgement = predict.copy().tolist()
    for i in range(len(judgement)):
        if confidence[i] <= threshold:
            judgement[i] = null_type

    return np.array(judgement)


def cm_with_percentage(y, pred, labels):
    n = len(labels)
    cm = np.zeros((n+1, n+1))
    cm[:-1, :-1] = confusion_matrix(y, pred, labels)

    cnt = 0
    for i in range(n):
        recall = cm[i, i] / np.sum(cm[i, :-1])
        cm[i, n] = round(recall * 100, 1)
        precision = cm[i, i] / np.sum(cm[:-1, i])
        cm[n, i] = round(precision * 100, 1)
        cnt += cm[i, i]

    accuracy = cnt / np.sum(cm[:-1, :-1])
    cm[n, n] = round(accuracy * 100, 1)

    return cm

def make_cm_labels(cm):
    n = len(cm) - 1
    anot = np.empty((n+1, n+1), dtype=object)

    for i in range(n):
        for j in range(n):
            anot[i][j] = int(cm[i][j])

    cnt = 0
    for i in range(n):
        if cm[i, n] >= 0:
            anot[i, n] = '{:.1f}%'.format(cm[i, n])
        else:
            print(cm[i,n])
        if cm[n, i] >= 0:
            anot[n, i] = '{:.1f}%'.format(cm[n, i])
        cnt += cm[i, i]

    anot[n, n] = '{:.1f}%'.format(cm[n, n])

    return anot

import base64

def data_url_to_image(data_url, image):
    prefix, data = data_url.split(',')

    with open(image, 'wb') as f:
        f.write(base64.b64decode(data))

def image_to_data_url(image):
    ext = image.split('.')[-1]

    prefix = 'data:image/' + ext + ';base64,'

    with open(image, 'rb') as f:
        data_url = prefix + base64.b64encode(f.read()).decode()

    return data_url

def data_url_to_pdf(data_url, file):
    prefix, data = data_url.split(',')

    with open(file, 'wb') as f:
        f.write(base64.b64decode(data))

def pdf_to_data_url(file):
    prefix = 'data:application/pdf;base64,'

    with open(file, 'rb') as f:
        data_url = prefix + base64.b64encode(f.read()).decode()

    return data_url

def draw_confusion_matrix(confusion_matrix, class_names, figsize = (20,18), dpi=40, cmap='GnBu'):
    sns.set(font_scale=2)
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    try:
        heatmap = sns.heatmap(df_cm, annot=make_cm_labels(confusion_matrix), fmt="",
                              square=True, cmap=cmap, linewidth=.5)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")



    heatmap.xaxis.tick_top()
    heatmap.xaxis.set_label_position('top')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.set_ylabel(ylabel='True label', labelpad=25)
    heatmap.set_xlabel(xlabel='Predicted label', labelpad=25)

    left  = 0.1  # the left side of the subplots of the figure
    right = 1    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.2   # the amount of height reserved for white space between subplots
    fig.subplots_adjust(left, bottom, right, top, wspace, hspace)
    # fig.tight_layout()
    return fig

# a = image_to_data_url('image.png')
# print(a)
# data_url_to_image(a, 'nhan-id-17.png')
# a = pdf_to_data_url('image.pdf')
# print(a)
# data_url_to_pdf(a, 'nhan-id-17.pdf')
