import numpy as np
import tensorflow as tf

def create_weights(shape, name=None):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(lenght, name=None):
	return tf.Variable(tf.constant(0.05, shape=[lenght]))

def create_conv_layer(input, num_input_channels, filter_size, num_filters, name=None, cnn_stride=1):
	with tf.variable_scope(name) as scope:
		shape = [filter_size, filter_size, num_input_channels, num_filters]

		weights = create_weights(shape, 'weights')

		biases = create_biases(num_filters, 'biases')

		layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, cnn_stride, cnn_stride, 1], padding='SAME', name='conv2d')
		layer += biases

		layer = tf.nn.relu(layer)

		layer = tf.nn.relu(layer, name='relu')
		#tf.summary.histogram("weights", weights)
		#tf.summary.histogram("biases", biases)
		#tf.summary.histogram("activations", layer)

		return layer, weights

def create_RNN(input, rnn_size, num_outputs, keep_prob, name='rnn'):
	with tf.variable_scope(name) as scope:
		#input = tf.transpose(input, [1,0,2])
		print(input)
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
		outputs, states = tf.nn.dynamic_rnn(lstm_cell, input, dtype=tf.float32)

		weights = create_weights([rnn_size, num_outputs], 'weights')
		biases = create_biases(num_outputs, 'biases')

		layer = tf.matmul(outputs[-1], weights) + biases

		return layer

def create_multi_layer_RNN(input, rnn_size, num_outputs, num_layers, keep_prob, name='rnn'):
	with tf.variable_scope(name) as scope:
		#input = tf.transpose(input, [1,0,2])
		lstm_cell = tf.nn.rnn.BasicLSTMCell(rnn_size, name='lstm_cell')
		lstm_cell = tf.nn.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
		outputs, states = tf.nn.static_rnn(lstm_cell, input, dtype=tf.float32)

		weights = create_weights([rnn_size, num_outputs], 'weights')
		biases = create_biases(num_outputs, 'biases')

		layer = tf.matmul(outputs[-1], weights) + biases

		return layer




def create_fully_connected_layer(input, num_inputs, num_outputs, relu=True, name=None):
	with tf.variable_scope(name) as scope:
		weights = create_weights([num_inputs, num_outputs], name='weights')
		biases = create_biases(num_outputs, name='biases')

		layer = tf.matmul(input, weights) + biases

		if relu:
			layer = tf.nn.relu(layer, name='relu')

		return layer

def local_response_normalization(input, name=None):
	return tf.nn.local_response_normalization(input, name=name)

def dropout(input, keep_prob=0.5, name=None):
	return tf.nn.dropout(input, keep_prob, name=name)

def pooling(input, pool_ksize=2, pool_stride=2, name=None):
	return tf.nn.max_pool(input, [1, pool_ksize, pool_ksize, 1], [1, pool_stride, pool_stride, 1], padding='SAME', name=name)

def flatten_layer(layer, name=None):
	with tf.variable_scope(name) as scope:
		layer_shape = layer.get_shape()
		num_features = layer_shape[1:4].num_elements()
		layer_flat = tf.reshape(layer, [-1, num_features])
		return layer_flat, num_features
