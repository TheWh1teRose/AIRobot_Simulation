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
import VGG_modells
import keras
import os.path
import gc

path = 'C:/Users/Paperspace/Documents/GitHub/AIRobot_Simulation/DataProcessing/traindata/pre_diff4_60/*'
file = glob.glob(path)
data = None
print(file)


for f in file:
	if data is None:
		print("loaded: " + f)
		data = pickle.load(open(f, "rb"))
		gc.collect()
		x_min = data[0].min(axis=(1,2), keepdims=True)
		x_max = data[0].max(axis=(1,2), keepdims=True)
		new_data = (data[0] - x_min)/(x_max - x_min)
		X[0] = new_data
		y[0] = data[1]
	else:
		data = pickle.load(open(f, "rb"))
		gc.collect()
		print("loaded: " + f)
		x_min = data[0].min(axis=(1,2), keepdims=True)
		x_max = data[0].max(axis=(1,2), keepdims=True)
		new_data = (data[0] - x_min)/(x_max - x_min)
		X.append(new_data)
		y.append(data[1])

print(len(X))
split_proc = 0.2
index_spliter = round(len(X) * split_proc)
X_train = X[index_spliter:]
X_test = X[:index_spliter]
y_train = y[index_spliter:]
y_test = y[:index_spliter]

image_width = 300
image_height = 300
image_depth = 6
num_lable = 8
batch_size = 8
num_epochs = 1000

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU': 56} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

checkpointer = keras.callbacks.ModelCheckpoint(filepath=os.path.join('ckpts', 'LRCN' + '_{epoch:02d}_{val_loss:.2f}.hdf5'), verbose=1, save_best_only=True)
tb = keras.callbacks.TensorBoard(log_dir=os.path.join('statistics', 'LRCN'))
early_stopper = keras.callbacks.EarlyStopping(patience=7)
timestamp = time.time()
csv_logger = keras.callbacks.CSVLogger(os.path.join('logs', 'LRCN' + '-' + 'training-' + str(timestamp) + '.log'))

model = VGG_modells.VGG_A(0.5, 7)
optimizer = keras.optimizers.Adam(lr=1e-5, decay=1e-6)
metrics = ['accuracy']
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

for range(100):
	for i in in range(len(X_train)):
		model.fit(X_train[i],
			y_train[i],
			batch_size=batch_size,
			verbose=2,
			epochs=1, shuffle=False)
		model.reset_states()

		score = []
		for j in range(len(X_test)):
			scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
			model.reset_states()
			score.append(scores[1])
		print("Model Accuracy: %.2f%%" % ((reduce(lambda x, y: x + y, score) / len(score))*100))
