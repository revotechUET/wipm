import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


def create_table_index(dim , tau, m):
	"""
	Arguments:
	m-- length of time series

	Returns:
	indx-- the indx of each vector
	"""
	num_vectors = m - (dim - 1)*tau
	indx = np.zeros((num_vectors, dim))
	indx[0,:] = np.arange(0, dim*tau, tau).astype(int)

	for i in range(1,num_vectors):
		indx[i,:] = indx[i-1,:]+1

	return indx.ravel()

def filter(data_well, label_well, label, dim ,tau):
	length = []
	vector_data_well_label = []
	index = np.where(label_well == label)[0]
	b = index[0]
	e = index[0]

	for i in range(len(index)-1):
		# khong lien tiep
		if(index[i+1] - index[i] != 1):
			if((e + 1 - b) > ((dim-1)*tau)):
				vector_data_well_label.append(data_well[b:e+1, 1:])

			# statistic length of sub timeseries
			if e+1-b > 0:
				length.append(e+1-b)

			b = index[i+1]
		# lien tiep
		else:
			e = index[i+1]

	if((e + 1 - b) > ((dim-1)*tau)):
		vector_data_well_label.append(data_well[b:e+1, 1:])
	# print('vector_data_well_label.shape: ', vector_data_well_label[0].shape)

	# # compute length of each sub timeseries
	# print('length: ', sorted(Counter(length).items(), key=lambda i: i[0]))

	return vector_data_well_label

def get_vector_each_vector_timeseries(vector_data_well_label, dim, tau):
	# vector train for feature in column 1
	timeseries_train = np.array(vector_data_well_label[:, 0], copy=True)

	indx_vectors_timeseries_train = create_table_index(dim, tau, timeseries_train.shape[0]).astype(int)
	vectors_train = timeseries_train[indx_vectors_timeseries_train].reshape((timeseries_train.shape[0] - (dim -1)*tau, dim))
	#vector train for another feature
	for i in range(1, vector_data_well_label.shape[1]):
		timeseries_train = np.array(vector_data_well_label[:, i], copy=True)
		vectors_train = np.concatenate((vectors_train, timeseries_train[indx_vectors_timeseries_train].reshape((timeseries_train.shape[0] - (dim -1)*tau, dim))), axis=1)
	return vectors_train

def extract_vector_train_each_well(X, y, dim, tau, label, train_well):
	data_well = X[np.where(X[:, 0] == train_well)[0], :]

	'''________________________________________________'''

	'''________________________________________________'''
	label_well = y[np.where(X[:, 0] == train_well)[0]]

	vector_data_well_label = filter(data_well, label_well, label, dim, tau)


	"""______vector train for first vector_____"""
	vectors_train = get_vector_each_vector_timeseries(vector_data_well_label[0], dim, tau)
	"""______vector train for another vector_____"""

	if(len(vector_data_well_label) >=2):
		for i in range(1, len(vector_data_well_label)):
			vectors_train = np.concatenate((vectors_train, get_vector_each_vector_timeseries(vector_data_well_label[i], dim, tau)))

	return vectors_train


def extract_vector_test(X, dim, tau):
	data_well_label = X

	timeseries_test = np.array(data_well_label[:, 1], copy=True)
	#_________________#
	# print('timeseries_test.shape: ', timeseries_test.shape)
	#_________________#
	indx_vectors_timeseries_test = create_table_index(dim, tau, timeseries_test.shape[0]).astype(int)
	vectors_test = timeseries_test[indx_vectors_timeseries_test].reshape((timeseries_test.shape[0] - (dim -1)*tau, dim))

	# # vector train for other feature
	# for i in range(2, X.shape[1]):
	# 	timeseries_test = np.array(data_well_label[:, i], copy=True)
	# 	vectors_test = np.concatenate((vectors_test, timeseries_test[indx_vectors_timeseries_test].reshape((timeseries_test.shape[0] - (dim -1)*tau, dim))), axis=1)

	return vectors_test, indx_vectors_timeseries_test


def minMaxScalerPreprocessing(X, minScaler = 0.0, maxScaler = 253):
	return (X - minScaler)/(maxScaler - minScaler)


def create_train(training_well_ts, dim, tau, curve_number, facies_class_number):
	# from sklearn.preprocessing import LabelEncoder
	# label_enc = LabelEncoder()
	# training_well_ts[:,0] = label_enc.fit_transform(training_well_ts[:, 0])

	#__debug__#
	# print('training_well_ts: ', training_well_ts.shape)
	# print('training_well_ts: ', training_well_ts[:, [curve_number]].shape)
	#_________#
	from sklearn.preprocessing import MinMaxScaler
	minMaxScaler = MinMaxScaler()
	#__debug___#
	# print(training_well_ts[:, [curve_number]].shape)
	#__________#
	training_well_ts[:, curve_number] = minMaxScaler.fit_transform(training_well_ts[:, [curve_number]]).ravel()
	X = training_well_ts[:, [0, curve_number]]
	y = training_well_ts[:, -1].astype(int)
	# print(type(y))
	train_well = list(set(X[:, 0]))

	# print('y: ')
	# print(y)
	# print('train_well', train_well)

	""" vector train of train_well[0]"""
	vectors_train = extract_vector_train_each_well(X, y, dim, tau, facies_class_number, train_well[0])
	"""_____________"""
	""" vector train for another train well"""
	if(len(train_well) > 1):
		for i in range(1, len(train_well)):
			vectors_train = np.concatenate((vectors_train, extract_vector_train_each_well(X, y, dim, tau, facies_class_number, train_well[i])))

	"""___________________________________________"""

	return vectors_train




def create_test(testing_well_ts, dim, tau, curve_number):

	# from sklearn.preprocessing import LabelEncoder
	# label_enc = LabelEncoder()
	# testing_well_ts[:,0] = label_enc.fit_transform(testing_well_ts[:, 0])
	from sklearn.preprocessing import MinMaxScaler
	minMaxScaler = MinMaxScaler()
	testing_well_ts[:, curve_number] = minMaxScaler.fit_transform(testing_well_ts[:, [curve_number]]).ravel()
	X = testing_well_ts[:, [0, curve_number]]
	y = testing_well_ts[:, -1].astype(int)
	#____________#
	# print('X_test: ', X)
	#____________#
	return extract_vector_test(X, dim, tau)


def train(training_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number, filename):
	vectors_train = create_train(training_well_ts, dim, tau, curve_number, facies_class_number)
	#__debug__#
	# print('vectors_train: ', vectors_train)
	# print('vectors_train.shape: ', vectors_train.shape)
	#_________# 
	np.savez_compressed(filename, train_data=vectors_train, dim=dim, tau=tau, epsilon=epsilon, lambd=lambd, percent=percent, curve_number=curve_number, facies_class_number=facies_class_number)

def predict(testing_well_ts, filename):
	data = np.load(filename)
	vectors_train = data['train_data']
	dim = data['dim']
	tau = data['tau']
	percent = data['percent']
	curve_number = data['curve_number']
	epsilon = data['epsilon']
	lambd = data['lambd']



	vectors_test, indx_vectors_timeseries_test = create_test(testing_well_ts, dim, tau, curve_number)
	r_dist = cdist(vectors_train, vectors_test, 'minkowski', p=1)

	#_________________#
	print('vectors_train.shape: ', vectors_train.shape)
	print('vectors_test.shape: ', vectors_test.shape)
	print('r_dist: ', r_dist.shape)
	print('min_r_dist: ', r_dist.min())
	print('r.max: ', r_dist.max())
	#_________________#

	r = np.sum(r_dist < epsilon, axis=0)
	#__debug__#
	# print(r)
	#_________#

# 	"""____________________________________________________"""

	predict_label = np.zeros((testing_well_ts[:, -1].shape[0], ), dtype=int)
	indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))
	# print('indx_vectors_timeseries_test.shape: ', indx_vectors_timeseries_test)

	index = indx_vectors_timeseries_test[r > lambd, :]
	index = index[:, 0]
	add_index = list(np.arange(0, dim*percent, dtype=int))

#	index = index[:, :int(dim*percent)].ravel()
	for i in add_index:
		predict_label[(index+i).ravel()] = 1

	return predict_label.ravel().tolist()

def get_data_from_json(data_json):
	trainset = data_json['train']

	label_enc = LabelEncoder()
	well = label_enc.fit_transform(np.array(trainset['well']).reshape(-1, 1)).reshape(-1, 1)
	trainset['data'] = np.array(trainset['data']).T
	trainset['target'] = np.array(trainset['target']).reshape(-1, 1)
	training_well_ts = np.concatenate((well, trainset['data'], trainset['target']), axis=1)

	params = data_json['params']
	dim = params['dim']
	if dim == None:
		raise AssertionError

	tau = params['tau']
	if tau == None:
		raise AssertionError

	epsilon = params['epsilon']
	if epsilon == None:
		raise AssertionError

	lambd = params['lambd']
	if lambd == None:
		raise AssertionError

	percent = params['percent']
	if percent == None:
		raise AssertionError

	curve_number = params['curve_number']
	if curve_number == None:
		raise AssertionError

	facies_class_number = params['facies_class_number']
	if facies_class_number == None:
		raise AssertionError

	return training_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number, data_json['model_id']

def load_dataset(file_name='../data/data.csv'):
	# load dataset
	data = pd.read_csv(file_name)

	X = data.iloc[:, :].values
	from sklearn.preprocessing import LabelEncoder
	label_enc = LabelEncoder()
	X[:, 0] = label_enc.fit_transform(X[:, 0])

	return X

def split_train_test(data, train_well=[0, 1, 2, 3, 4, 5, 6], test_well=7):
	train = data[data[:, 0] == train_well[0], :]
	if len(train_well) >= 2:
		for i in range(1, len(train_well)):
			train = np.concatenate((train, data[data[:, 0] == train_well[i], :]), axis = 0)
	#__debug__#
	# print('train: ', train)
	#_________#
	test = data[data[:, 0] == test_well, :]

	return train, test

def swap_all(dim, tau, epsilon, lambd, percent, curve_number, facies_class_number):
	X = load_dataset()


	training_well_ts, testing_well_ts = split_train_test(X)
	#__debug__#
	# print('training_well_ts: ', training_well_ts.shape)
	# print(training_well_ts)
	#_________#
	y_test = (testing_well_ts[:, -1] == facies_class_number).astype(int)


	train(training_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number, id_string=str(facies_class_number))
	target = predict(testing_well_ts, id_string=str(facies_class_number))
	target = np.array(target)
	#__debug__#
	# print('target.shape: ', target)
	# print('y_test.shape: ', y_test)
	#_________#
	print('num predict: ', np.sum(target))
	print('accuracy: ', np.mean(target==y_test))
	print(classification_report(y_test, target))
	print(confusion_matrix(y_test, target))
	return target


def main():

	dim = 4
	tau = 2
	epsilon = [0, 0, 0, 0.035, 0.1, 0.02, 0.035, 0.05, 0.025]
	lambd = 20
	percent = 1
	curve_number = 7
	facies_class_number = 5
	# training_well_ts, testing_well_ts = get_data()
	# # print(training_well_ts)
	# # print(testing_well_ts)
	# predict_vector = rqa(training_well_ts, testing_well_ts, dim=dim, tau=tau, epsilon=epsilon, lambd=lambd, percent=percent, curve_number=1, facies_class_number=5)
	# import json
	# data = json.load(open('data.json'))
	# training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number = get_data_from_json(data)
	all_target = []
	for curve_number in [3, 5, 7, 8]:
		all_target.append(swap_all(dim, tau, epsilon[curve_number], lambd, percent, curve_number, facies_class_number))

if __name__ == '__main__':
	main()
