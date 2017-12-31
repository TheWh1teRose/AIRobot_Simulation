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
import VGG_modells as modell

path = 'C:/Users/Paperspace/Documents/GitHub/AIRobot_Simulation/DataProcessing/traindata/pre_diff3_full/*'
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


#X = X[:,:,:,:3]
print(X.shape)

x_min = X.min(axis=(1,2), keepdims=True)
x_max = X.max(axis=(1,2), keepdims=True)
X = (X - x_min)/(x_max - x_min)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_test, X_valid = np.array_split(X_test, 2)
y_test, y_valid = np.array_split(y_test, 2)
print("X train: " + str(X_train.shape))
print("y train: " + str(y_train.shape))
print("X test: " + str(X_test.shape))
print("y test: " + str(y_test.shape))
print("X valid: " + str(X_valid.shape))
print("y valid: " + str(y_valid.shape))

print("sample: ")
print(X_train[2,:10,:10,1])
print(y_train[2])

image_width = 300
image_height = 300
image_depth = 6
num_lable = 8

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
for modellID in range(2):
	for lr in [1E-4]:
		graph = tf.Graph()
		with graph.as_default():
			tf_X_train = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_depth], name='X')
			tf_y_train = tf.placeholder(tf.float32, shape=[None, num_lable], name='y')
			tf_y_train_cls = tf.argmax(tf_y_train, dimension=1)
			keep_prob = tf.placeholder(tf.float32, name='keep_prob')
			tf_images = tf.image.resize_images(tf_X_train, [224, 224])

			global_step = tf.Variable(0, trainable=False)
			learning_rate = tf.train.exponential_decay(lr, global_step, 1000, 0.96, staircase=True)

			if modellID == 0:
				logits = modell.VGG_D(keep_prob, tf_images)
			if modellID == 1:
				logits = modell.VGG_E(keep_prob, tf_images)
			if modellID == 2:
				logits = modell.VGG_E(keep_prob, tf_images)


			y_pred = tf.nn.softmax(logits)
			y_pred_cls = tf.argmax(y_pred, dimension=1)
			cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_y_train)
			cost = tf.reduce_mean(cross_entropy)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
			correct_prediction = tf.equal(y_pred_cls, tf_y_train_cls)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			tf.summary.scalar('Accuracy', accuracy)
			tf.summary.scalar('loss/cost', cost)
			tf.summary.scalar('learning_rate', learning_rate)

			merged = tf.summary.merge_all()
			saver = tf.train.Saver()


		tf.reset_default_graph()

		with tf.Session(graph=graph) as session:
			tf.global_variables_initializer().run()
			train_writer = tf.summary.FileWriter('statistics/summ_Modell{}_{}'.format(modellID+4, lr), session.graph)
			total_epochs = 1
			courent_batch_position = 0
			epochs = 1000
			batch_size = 32
			keep_probability = 0.5

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
				summary, acc = session.run([merged, accuracy], feed_dict={tf_X_train: X_batch, tf_y_train: y_batch, keep_prob: 1})
				run_metadata = tf.RunMetadata()
				train_writer.add_run_metadata(run_metadata, 'step{}'.format(total_epochs))
				train_writer.add_summary(summary, total_epochs)
				total_epochs += 1
				if total_epochs % 100 == 0:

					all_acc_valid = []
					courent_batch_position_valid = 0
					for i in range(int(X_valid.shape[0]/batch_size)+1):
						if (courent_batch_position_valid + batch_size) <= X_valid.shape[0]:
							nextBatchPosition_valid = (courent_batch_position_valid+batch_size)
							X_batch_valid = X_valid[courent_batch_position_valid : nextBatchPosition_valid,...]
							y_batch_valid = y_valid[courent_batch_position_valid : nextBatchPosition_valid,...]
							courent_batch_position_valid += batch_size
						else:
							overLapp_valid = ((courent_batch_position_valid + batch_size) - X_valid.shape[0])
							X_batch_valid = X_valid[: overLapp_valid,...]
							y_batch_valid = y_valid[: overLapp_valid,...]
							courent_batch_position_valid = overLapp_valid

							all_acc_valid.append(session.run(accuracy, feed_dict={tf_X_train: X_batch_valid, tf_y_train:y_batch_valid, keep_prob: 1}))
					tf.summary.scalar('Validation_Accuracy', functools.reduce(lambda x_, y_: x_ + y_, all_acc_valid)/len(all_acc_valid))

					print("Valid Accuracy: " + str(functools.reduce(lambda x_, y_: x_ + y_, all_acc_valid)/len(all_acc_valid)))
					print("loss: " + str())

					print("epoch: {}; Train Accuracy: {}".format(total_epochs, acc))
					print("batches seen: " + str(tf.train.global_step(session, global_step)))

				if total_epochs % 250 == 0:
					save_path = saver.save(session, "ckpts/model_{}_{}_{}.ckpt".format(total_epochs, modellID+4, lr))
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

			all_acc_test.append(session.run(accuracy, feed_dict={tf_X_train: X_batch, tf_y_train:y_batch, keep_prob: 1}))
			print("Test Accuracy: " + str(functools.reduce(lambda x, y: x + y, all_acc_test)/len(all_acc_test)))
			session.close()
