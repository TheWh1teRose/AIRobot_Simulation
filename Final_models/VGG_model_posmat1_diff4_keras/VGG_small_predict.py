import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import glob
import CNN_utils as cnn
import pickle
import time
import datetime
import functools
from PIL import ImageGrab
import cv2
from Controller import Controller
from threading import Thread
import keras

def updatePositionMatrix():
	global lastTime
	global posMatrix
	cntRcv = Controller("127.0.0.1", 5002)
	cntRcv.startController()

	smoothing = 3

	while True:
		UDPData, address = cntRcv.recvData()
		controlsStr = UDPData.decode("utf-8").split("$")[0]
		isRestarted = int(UDPData.decode("utf-8").split("$")[1])
		posInMatrixStr = UDPData.decode("utf-8").split("$")[2]
		posInMatrix = list(map(int, posInMatrixStr.split(":")))
		if posInMatrix[0] < posMatrixSize[0] and posInMatrix[1] < posMatrixSize[1] and posInMatrix[1] < posMatrixSize[1]:
			posMatrix[posInMatrix[0], posInMatrix[1], posInMatrix[2]] = 1

		#decreasing the values over time with smoothing
		deltaTime = time.time() - lastTime
		lastTime = time.time()
		posMatrix = posMatrix - (deltaTime/smoothing)
		posMatrix = posMatrix.clip(min=0)

def getPositionMatrixImages():
	posMatrixSumX = np.sum(posMatrix, axis=0).repeat(5,axis=0).repeat(5,axis=1)
	posMatrixSumY = np.sum(posMatrix, axis=1).repeat(5,axis=0).repeat(5,axis=1)
	posMatrixSumZ = np.sum(posMatrix, axis=2).repeat(5,axis=0).repeat(5,axis=1)
	return posMatrixSumX, posMatrixSumY, posMatrixSumZ



model = keras.models.load_model('ckpts/CNN_50_0.38.hdf5')
cnt = Controller("127.0.0.1", 5003)
#cnt.startController()

controlsStr = "0:0:0:0"
posInMatrixStr = "0:0:0"
isRestarted = 0;
posMatrixSize = [60,60,60]
posMatrix = np.zeros((posMatrixSize[0],posMatrixSize[1],posMatrixSize[2]))
lastTime = time.time()

posMatThread = Thread(target = updatePositionMatrix, args = [])
posMatThread.start()

while True:
	data = None
	cls_predection = None
	printscreen = np.array(ImageGrab.grab(bbox=(2,50,302,350)))

	posMatrixSumX, posMatrixSumY, posMatrixSumZ = getPositionMatrixImages()
	x = np.stack((printscreen[:,:,0], printscreen[:,:,1], printscreen[:,:,2], posMatrixSumX, posMatrixSumY, posMatrixSumZ), axis=2)

	data = x[np.newaxis,...]
	cv2.imshow('screen', np.array(data[0,:,:,:3],dtype=np.int8))
	cv2.imshow('posMatrixX', data[0,:,:,3])
	cv2.imshow('posMatrixY', data[0,:,:,4])
	cv2.imshow('posMatrixZ', data[0,:,:,5])
	data_min = data.min(axis=(1,2), keepdims=True)
	data_max = data.max(axis=(1,2), keepdims=True)
	data = (data - data_min)/(data_max - data_min)

	prediction = model.predict(data)
	predicted_data = prediction[0]
	prediction = np.unravel_index(predicted_data.argmax(), predicted_data.shape)
	cls_predection = np.zeros(8, dtype=np.int8)
	cls_predection[prediction] = 1

	send_data = ''
	for p in cls_predection:
		send_data = send_data + str(p) + ':'
	send_data = send_data[:-1]
	print(send_data)
	cnt.sendMessage(send_data.encode('utf-8'))
	#time.sleep(0.15)
	if cv2.waitKey(25) & 0xFF == ord('q'): #quit statement
		cv2.destroyAllWindows()
		break
