import numpy as np
import glob
import sys
import gc
import time
import datetime
import pandas as pd
import preprocessingUtils as pu
import pickle

gc.enable()

maxMemoryUsage = 10000
path = 'traindata/raw/*.npy'
file = glob.glob(path)
data = np.array([np.array((300,300,6), dtype=np.float64), np.array(8, dtype=np.int8)])

def saveData(data):
	#delete the first placeholder column
	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')
	pickle.dump(data, open("traindata/pre/data_1_"+st, "wb"))
	print('saved data to disk')

def processData(data):
	data = pu.dropData(np.array([0,0,0,0,0,0,0,0]), data)
	data = np.delete(data, [0,1,2], 0)
	data = pu.dropData(np.array([0,0,0,0,0,0,1,0]), data)
	data = np.delete(data, [0,1,2], 0)
	data = pu.dropData(np.array([0,1,0,0,0,0,0,1]), data)
	data = np.delete(data, [0,1,2], 0)
	data = pu.dropData(np.array([0,0,0,0,0,0,0,1]), data)
	data = np.delete(data, [0,1,2], 0)
	data = pu.dropData(np.array([0,0,0,0,1,0,1,0]), data)
	data = np.delete(data, [0,1,2], 0)
	gc.collect()
	#data = np.random.shuffle(data)
	print(data[:,1].shape)
	full_X = data[:,0]
	full_Y = data[:,1]
	new_X = full_X[0][np.newaxis,...]
	new_Y = full_Y[0][np.newaxis,...]
	for i in range(full_Y.shape[0]-1):
		new_Y = np.concatenate((new_Y, full_Y[i+1][np.newaxis,...]), axis=0)
		new_X = np.concatenate((new_X, full_X[i][np.newaxis,...]), axis=0)

	split_x = np.array_split(new_X, new_X.shape[0]/7)
	new_X = None
	gc.collect()
	X = None
	for element in split_x:
		if element.shape[0] != 7:
			continue
		if X is None:
			X = element[np.newaxis,...]
		else:
			print(element.shape)
			print(X.shape)
			X = np.concatenate((X,element[np.newaxis,...]), axis=0)

	split_y = np.array_split(new_Y, new_Y.shape[0]/7)
	new_Y = None
	gc.collect()
	y = None
	for element in split_y:
		if element.shape[0] != 7:
			continue
		if y is None:
			y = element[np.newaxis,...]
		else:
			print(element.shape)
			print(y.shape)
			y = np.concatenate((y,element[np.newaxis,...]), axis=0)


	new_data = [np.array(X), np.array(y)]
	#new_data = pu.qualifyData(new_data)
	print(new_data[0].shape)
	print(new_data[1].shape)
	return new_data
	for i in range(full_X.shape[0]):
		if i==0:
			addArray = full_X[i]
			new_X = addArray[np.newaxis,...]
		if i==1:
			addArray = full_X[i]
			print("test")
			print(addArray.shape)
			print(new_X.shape)
			new_X = np.concatenate((new_X, addArray[np.newaxis,...]), axis=0)
		if i > 1:
			addArray = full_X[i]
			print("test")
			print(new_X.shape)
			new_X = np.concatenate((new_X, addArray[np.newaxis,...]), axis=0)
	for i in range(full_Y.shape[0]):
		if i==0:
			addArray = full_Y[i]
			new_Y = addArray[np.newaxis,...]
		if i==1:
			addArray = full_Y[i]
			print("test")
			print(addArray.shape)
			print(new_Y.shape)
			new_Y = np.concatenate((new_Y, addArray[np.newaxis,...]), axis=0)
		if i > 1:
			addArray = full_Y[i]
			print("test")
			print(new_Y.shape)
			new_Y = np.concatenate((new_Y, addArray[np.newaxis,...]), axis=0)


for f in file:
	lastoldData = np.load(f)
	print(lastoldData.shape)
	lastoldData = np.delete(lastoldData, 0,0)
	data = np.vstack((data, lastoldData))
	print("load: " + f)
	if sys.getsizeof(data)>maxMemoryUsage:
		data = processData(data)
		saveData(data)
		data = np.array([np.array((300,300,6), dtype=np.float64), np.array(8, dtype=np.int8)])
		gc.collect()

pu.qualifyData(data)
data = processData(data)
saveData(data)
