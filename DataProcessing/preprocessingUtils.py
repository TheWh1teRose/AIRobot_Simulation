import numpy as np 
import threading
import glob
import sys
import gc
import time
import datetime
import pandas as pd

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

def qualifyData(oldData):
	stringData = []
	for i in range(oldData.shape[0]):
		stringData.append(convertByteLikeArrayToString(oldData[i][1]))

	dataframe = pd.DataFrame({'data':stringData})
	print(dataframe['data'].value_counts()[2])

	minValue = dataframe['data'].value_counts().min()
	differentValues = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0], [1,0,0,0,0,0,1,0]]

	

	gc.collect()

def convertByteLikeArrayToString(data):
		stringData = ""
		for place in np.nditer(data):
			if place > 0:
				stringData += "1"
			else:
				stringData += "0"
		return stringData
