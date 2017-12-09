import numpy as np
import tensorflow as tf

def create_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(lenght):
	return tf.Variable(tf.constant(0.05, shape=[lenght]))

def create_conv_layer(input, num_input_channels, filter_size, num_filters, cnn_stride=1):
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	weights = create_weights(shape)
	print("test")
	print(weights)
	biases = create_biases(num_filters)
	print(input)
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, cnn_stride, cnn_stride, 1], padding='SAME')
	layer += biases

	layer = tf.nn.relu(layer)

	return layer, weights

def create_fully_connected_layer(input, num_inputs, num_outputs, relu=True):
	weights = create_weights([num_inputs, num_outputs])
	biases = create_biases(num_outputs)

	layer = tf.matmul(input, weights) + biases

	if relu:
		layer = tf.nn.relu(layer)

	return layer

def local_response_normalization(input):
	return tf.nn.local_response_normalization(input)

def dropout(input, keep_prob=0.5):
	return tf.nn.dropout(input, keep_prob)

def pooling(input, pool_ksize=2, pool_stride=2):
	return tf.nn.max_pool(input, [1, pool_ksize, pool_ksize, 1], [1, pool_stride, pool_stride, 1], padding='SAME')

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat, num_features

