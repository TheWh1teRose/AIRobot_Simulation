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

		layer = tf.nn.relu(layer, name='relu')

		return layer, weights

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
