import numpy as np
import threading
import glob
import sys
import gc
import time
import datetime
import pandas as pd
import sklearn

def dropData(dropArray, oldData):
	#default path 'traindata/raw/*.npy'
	data = np.array([np.zeros((300,300,6), dtype=np.float64), np.zeros(8, dtype=np.int8)])

	dropIndex = []
	for i in range(oldData.shape[0]):
		if np.array_equal(oldData[i][1], dropArray):
			dropIndex.append(i)

	oldData = np.delete(oldData, dropIndex, 0)
	data = np.vstack((data, oldData))
	print(data.shape)

	oldData = np.array([np.zeros((300,300,6), dtype=np.float64), np.zeros(8, dtype=np.int8)])
	gc.collect()

	return data

def qualifyData(data):
	stringData = []
	for i in range(data[0].shape[0]):
		stringData.append(convertByteLikeArrayToString(data[1][i]))

	dataframe = pd.DataFrame({'data':stringData})
	counts = dataframe['data'].value_counts().values
	min_counts = np.amin(counts)
	print(dataframe['data'].value_counts())

	minValue = dataframe['data'].value_counts().min()
	shuffled_X, shuffled_Y = sklearn.utils.shuffle(data[0], data[1])
	new_X = shuffled_X[0][np.newaxis,...]
	new_Y = shuffled_Y[0][np.newaxis,...]
	differentValues = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0]]

	for value in differentValues:
		tmp_X = np.zeros([1,300,300,6])
		tmp_Y = np.zeros([1,8])
		for i in range(shuffled_Y.shape[0]):
			if np.array_equal(np.array(shuffled_Y[i-1]), value):
				tmp_Y = np.concatenate((tmp_Y, shuffled_Y[i-1][np.newaxis,...]), axis=0)
				tmp_X = np.concatenate((tmp_X, shuffled_X[i-1][np.newaxis,...]), axis=0)
			if tmp_X.shape[0]==minValue+1:
				break
		tmp_X = np.delete(tmp_X, 0, 0)
		tmp_Y = np.delete(tmp_Y, 0, 0)

		new_X = np.concatenate((new_X, tmp_X), axis=0)
		print(new_Y.shape)
		print(tmp_Y.shape)
		new_Y = np.concatenate((new_Y, tmp_Y), axis=0)

	new_data = [new_X, new_Y]
	print(new_data[0].shape)
	return new_data

def convertByteLikeArrayToString(data):
		stringData = ""
		for place in np.nditer(data):
			if place > 0:
				stringData += "1"
			else:
				stringData += "0"
		return stringData
