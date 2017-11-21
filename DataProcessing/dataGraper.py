from Controller import Controller
import numpy as np
import time
import datetime
import sys
import gc
from PIL import ImageGrab
import cv2
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#enable garbage collector to free memory
gc.enable()

controlsStr = "0:0:0:0"
posInMatrixStr = "0:0:0"
isRestarted = 0;
posMatrixSize = [30,30,30]
posMatrix = np.zeros((posMatrixSize[0],posMatrixSize[1],posMatrixSize[2]))
lastTime = time.time()
data = np.array((np.array((300,300,6), dtype=np.float64), np.array(8, dtype=np.int8)))

#start the UDP connection to Unity that receive the controls an the position in the position matrix
cnt = Controller("127.0.0.1", 5002)

#define function to get the controls and position in matrix from the Unity UDP sender 
def getData():
	#get the controls from the unity Input.GetAxis function
	#in the format: [horizontal]:[vertical]:[height]:[graping]
	#and get the position in the matrix as [X]:[Y]:[Z]
	while True:
		global controlsStr
		global posInMatrixStr
		global isRestarted
		data, address = cnt.recvData()
		controlsStr = data.decode("utf-8").split("$")[0]
		isRestarted = int(data.decode("utf-8").split("$")[1])
		posInMatrixStr = data.decode("utf-8").split("$")[2]


#define function to convert the controls to an oneHot Array; -1:1:0:1 -> 0:1:1:0:0:0:1:0 
def convertControls(controls):
	beforeControls = controlsStr.split(":")
	oneHotControls = []
	for i, value in enumerate(beforeControls):
		value = float(value)
		if value >= 0:
			oneHotControls.append(value)
			oneHotControls.append(0)
		else: 
			oneHotControls.append(0)
			oneHotControls.append(abs(value))
	return np.array(oneHotControls)

#function to update the position-matrix
def updatePositionMatrix(smoothing):
	global lastTime
	global posMatrix
	posInMatrix = list(map(int, posInMatrixStr.split(":")))
	if posInMatrix[0] < posMatrixSize[0] and posInMatrix[1] < posMatrixSize[1] and posInMatrix[1] < posMatrixSize[1]:
		posMatrix[posInMatrix[0], posInMatrix[1], posInMatrix[2]] = 1

	#decreasing the values over time with smoothing
	deltaTime = time.time() - lastTime
	lastTime = time.time()
	posMatrix = posMatrix - (deltaTime/smoothing)
	posMatrix = posMatrix.clip(min=0)

#sum up the position-matrix fro the axis to visualizes the matrix
def getPositionMatrixImages():
	posMatrixSumX = np.sum(posMatrix, axis=0).repeat(10,axis=0).repeat(10,axis=1)
	posMatrixSumY = np.sum(posMatrix, axis=1).repeat(10,axis=0).repeat(10,axis=1)
	posMatrixSumZ = np.sum(posMatrix, axis=2).repeat(10,axis=0).repeat(10,axis=1)
	return posMatrixSumX, posMatrixSumY, posMatrixSumZ

#save the training data to disc
def saveData(data):
	data = np.delete(data, 0, 0)
	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')
	np.save("traindata/data_1_"+st ,data)
	print('saved data to disk')

#start the contollers
cnt.startController()
print("start Controller")

#start the getControls and getPosMAtrix function in a new thread to avoid data stagnation while reading image data
thread = threading.Thread(target=getData)
thread.start()

while True:
	#convert controls to onehot and print it
	controls = convertControls(controlsStr)

	updatePositionMatrix(3)
	posMatrixSumX, posMatrixSumY, posMatrixSumZ = getPositionMatrixImages()
	#read the screen, put it in to an numpy array and show it in a window
	printscreen = np.array(ImageGrab.grab(bbox=(2,50,302,350)))
	x = np.stack((printscreen[:,:,0], printscreen[:,:,1], printscreen[:,:,2], posMatrixSumX, posMatrixSumY, posMatrixSumZ), axis=2)
	y = controls
	data = np.vstack((data, np.array((x,y))))

	#checks if it is restarted then save it to disk and reset the position-matrix and the data array
	if sys.getsizeof(data)>500:
		threadSaveData = threading.Thread(target=saveData, args=(data,))
		threadSaveData.start()
		data = np.array((np.array((300,300,6), dtype=np.float64), np.array(8, dtype=np.int8)))
		posMatrix = np.zeros((posMatrixSize[0],posMatrixSize[1],posMatrixSize[2]))
		gc.collect()
	
	#show the position-matrix and the screen	
	cv2.imshow('screen', np.array(x[:,:,:3],dtype=np.int8))
	cv2.imshow('posMatrixX', x[:,:,3])
	cv2.imshow('posMatrixY', x[:,:,4])
	cv2.imshow('posMatrixZ', x[:,:,5])
	print(sys.getsizeof(data))
	if cv2.waitKey(25) & 0xFF == ord('q'): #quit statement 
            cv2.destroyAllWindows()
            break

