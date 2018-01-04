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
import VGG_modells as modell

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
	posMatrixSumX = np.sum(posMatrix, axis=0).repeat(10,axis=0).repeat(10,axis=1)
	posMatrixSumY = np.sum(posMatrix, axis=1).repeat(10,axis=0).repeat(10,axis=1)
	posMatrixSumZ = np.sum(posMatrix, axis=2).repeat(10,axis=0).repeat(10,axis=1)
	return posMatrixSumX, posMatrixSumY, posMatrixSumZ

#architecture
filter_size1 = 3
num_filters1 = 64
filter_size2 = 3
num_filters2 = 64
#maxpool
filter_size3 = 3
num_filters3 = 128
filter_size4 = 3
num_filters4 = 128
#maxpool
filter_size5 = 3
num_filters5 = 256
filter_size6 = 3
num_filters6 = 256
#maxpool
filter_size7 = 3
num_filters7 = 512
filter_size8 = 3
num_filters8 = 512
#maxpool
filter_size9 = 3
num_filters9 = 512
filter_size10 = 3
num_filters10 = 512
#maxpool
fc_size1 	 = 4096
fc_size2 	 = 4096
fc_size3 	 = 1000

image_width = 300
image_height = 300
image_depth = 6
num_lable = 8


epochs = 3000
start_learning_rate = 0.0001
batch_size = 16
keep_probability = 0.5

graph = tf.Graph()
with graph.as_default():
	tf_X_train = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_depth], name='X')
	tf_y_train = tf.placeholder(tf.float32, shape=[None, num_lable], name='y')
	tf_y_train_cls = tf.argmax(tf_y_train, dimension=1)
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 1000, 0.96, staircase=True)
	print(tf_X_train)

	tf_images = tf.image.resize_images(tf_X_train, [224, 224])

	logits = modell.VGG_A(keep_prob, tf_images)



	y_pred = tf.nn.softmax(logits)
	y_pred_cls = tf.argmax(y_pred, dimension=1)
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_y_train)
	cost = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
	correct_prediction = tf.equal(y_pred_cls, tf_y_train_cls)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	tf.summary.scalar('Accuracy', accuracy)
	tf.summary.scalar('loss/cost', cost)
	merged = tf.summary.merge_all()
	saver = tf.train.Saver()


tf.reset_default_graph()
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	saver.restore(session, 'ckpts/model_1000_1_kp0.5.ckpt')
	cnt = Controller("127.0.0.1", 5003)
	#cnt.startController()

	controlsStr = "0:0:0:0"
	posInMatrixStr = "0:0:0"
	isRestarted = 0;
	posMatrixSize = [30,30,30]
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

		prediction = session.run(y_pred_cls, feed_dict={tf_X_train: data, keep_prob:1.0})
		prediction_cls = session.run(y_pred, feed_dict={tf_X_train: data, keep_prob:1.0})
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


	session.close()
