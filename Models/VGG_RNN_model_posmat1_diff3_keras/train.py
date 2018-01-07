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

path = 'C:/Users/Paperspace/Documents/GitHub/AIRobot_Simulation/DataProcessing/traindata/pre_diff3_long/*'
file = glob.glob(path)
data = None
print(file)

for f in file:
	if data is None:
		print("loaded: " + f)
		data = pickle.load(open(f, "rb"))
		print("loaded: " + f)
		X = data[0]
		y = data[1]
	else:
		data = pickle.load(open(f, "rb"))
		X = np.concatenate((X, data[0]))
		y = np.concatenate((y, data[1]))

x_min = X.min(axis=(1,2), keepdims=True)
x_max = X.max(axis=(1,2), keepdims=True)
X = (X - x_min)/(x_max - x_min)

new_x = np.array_split(X, X.shape[0]/7)
X = None
for element in new_x[1:]:
	if element.shape[0] != 7:
		continue
	if X is None:
		X = element[np.newaxis,...]
	else:
		print(element.shape)
		print(X.shape)
		X = np.concatenate((X,element[np.newaxis,...]), axis=0)

new_y = np.array_split(y, y.shape[0]/7)
y = None
for element in new_y[1:]:
	if element.shape[0] != 7:
		continue
	if y is None:
		y = element[np.newaxis,...]
	else:
		print(element.shape)
		print(y.shape)
		y = np.concatenate((y,element[np.newaxis,...]), axis=0)

print(X.shape)
print(y.shape)

#X = X[:,:,:,:3]
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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

modell = keras.models.load_model('LRCN_17_0.38.hdf5')
optimizer = keras.optimizers.Adam(lr=1e-5, decay=1e-6)
metrics = ['accuracy']
modell.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
modell.fit(X_train,
	y_train,
	batch_size=batch_size,
	validation_data=(X_test, y_test),
	verbose=1,
	callbacks=[tb, early_stopper, csv_logger, checkpointer],
	epochs=num_epochs)
