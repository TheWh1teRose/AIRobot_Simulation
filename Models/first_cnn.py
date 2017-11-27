import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
import CNN_utils as cnn
import pickle
import time
import datetime

path = 'D:/Dokumente/Programmieren/RoboPen/UnitySimulation/AIRobot_Simulation/DataProcessing/traindata/pre/*'
file = glob.glob(path)
data = None

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


X = X[:,:,:,:3]
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

#architecture
filter_size1 = 3
num_filters1 = 64
#maxpool
filter_size3 = 3
num_filters3 = 128
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
image_depth = 3
num_lable = 8


epochs = 10000
learning_rate = 0.0001
batch_size = 1
keep_probability = 0.2

graph = tf.Graph()
with graph.as_default():
	tf_X_train = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_depth], name='X')
	tf_y_train = tf.placeholder(tf.float32, shape=[None, num_lable], name='y')
	tf_y_train_cls = tf.argmax(tf_y_train, dimension=1)
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	print(tf_X_train)

	layer_conv1, weights_conv1 = cnn.create_conv_layer(tf_X_train, image_depth, filter_size1, num_filters1)
	layer_conv1_pool = cnn.pooling(layer_conv1)

	layer_conv3, weights_conv3 = cnn.create_conv_layer(layer_conv1_pool, num_filters1, filter_size3, num_filters3)
	layer_conv3_pool = cnn.pooling(layer_conv3)

	layer_conv5, weights_conv5 = cnn.create_conv_layer(layer_conv3_pool, num_filters3, filter_size5, num_filters5)
	layer_conv6, weights_conv6 = cnn.create_conv_layer(layer_conv5, num_filters5, filter_size6, num_filters6)
	layer_conv6_pool = cnn.pooling(layer_conv6)

	layer_conv7, weights_conv7 = cnn.create_conv_layer(layer_conv6_pool, num_filters6, filter_size7, num_filters7)
	layer_conv8, weights_conv8 = cnn.create_conv_layer(layer_conv7, num_filters7, filter_size8, num_filters8)
	layer_conv8_pool = cnn.pooling(layer_conv8)

	layer_conv9, weights_conv9 = cnn.create_conv_layer(layer_conv8_pool, num_filters8, filter_size9, num_filters9)
	layer_conv10, weights_conv10 = cnn.create_conv_layer(layer_conv9, num_filters9, filter_size10, num_filters10)
	layer_conv10_pool = cnn.pooling(layer_conv10)


	layer_flat, num_features = cnn.flatten_layer(layer_conv10_pool)
	fc_layer1 = cnn.create_fully_connected_layer(layer_flat, num_features, fc_size1)
	fc_layer1_dropout = cnn.dropout(fc_layer1, keep_prob)
	fc_layer2 = cnn.create_fully_connected_layer(fc_layer1_dropout, fc_size1, fc_size2)
	fc_layer2_dropout = cnn.dropout(fc_layer2, keep_prob)
	fc_layer3 = cnn.create_fully_connected_layer(fc_layer2_dropout, fc_size2, fc_size3)
	fc_layer4 = cnn.create_fully_connected_layer(fc_layer3, fc_size3, num_lable, relu=False)


	y_pred = tf.nn.softmax(fc_layer4)
	y_pred_cls = tf.argmax(y_pred, dimension=1)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer4, labels=tf_y_train)
	cost = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	correct_prediction = tf.equal(y_pred_cls, tf_y_train_cls)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	tf.summary.scalar('Accuracy', accuracy)
	tf.summary.scalar('loss/cost', cost)
	merged = tf.summary.merge_all()



with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	train_writer = tf.summary.FileWriter('statistics/train', session.graph)

	total_epochs = 1
	courent_batch_position = 0
	start_time = time.time()

	for i in range(epochs):
		if (courent_batch_position + batch_size) <= X_train.shape[0]:
			nextBatchPosition = (courent_batch_position+batch_size)
			X_batch = X_train[courent_batch_position : nextBatchPosition,...]
			y_batch = y_train[courent_batch_position : nextBatchPosition,...]
			courent_batch_position += batch_size
		else:
			overLapp = ((courent_batch_position + batch_size) - X_train.shape[0])
			X_batch = X_train[: overLapp,...]
			y_batch = y_train[: overLapp,...]
			courent_batch_position = overLapp

		session.run(optimizer, feed_dict={tf_X_train: X_batch, tf_y_train: y_batch, keep_prob: keep_probability})
		total_epochs += 1
		if total_epochs % 100 == 0:
			summary, acc = session.run([merged, accuracy], feed_dict={tf_X_train: X_batch, tf_y_train: y_batch, keep_prob: 0})
			train_writer.add_summary(summary, total_epochs)
			print("epoch: {}; Train Accuracy: {}".format(total_epochs, acc))

	end_time = time.time()
	time_dif = end_time - start_time
	print("Used Time: " + str(datetime.timedelta(seconds=int(round(time_dif)))))

	all_acc_test = []
	for i in range(int(X_test.shape[0]/batch_size)+1):
		if (courent_batch_position + batch_size) <= X_test.shape[0]:
			nextBatchPosition = (courent_batch_position+batch_size)
			X_batch = X_test[courent_batch_position : nextBatchPosition,...]
			y_batch = y_test[courent_batch_position : nextBatchPosition,...]
			courent_batch_position += batch_size
		else:
			overLapp = ((courent_batch_position + batch_size) - X_test.shape[0])
			X_batch = X_test[: overLapp,...]
			y_batch = y_test[: overLapp,...]
			courent_batch_position = overLapp

		all_acc_test.append(session.run(accuracy, feed_dict={tf_X_train: X_batch, tf_y_train:y_batch, keep_prob: 0}))
		print("Test Accuracy: " + (reduce(lambda x, y: x + y, all_acc_test)/len(all_acc_test)))
