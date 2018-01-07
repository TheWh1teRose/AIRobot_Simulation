import numpy as np
import tensorflow as tf
import CNN_utils as cnn

def VGG_A(keep_prob, data):
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
    rnn_size1 = 4096


    image_width = 300
    image_height = 300
    image_depth = 6
    num_lable = 8

    keep_probability = 0.5


    with tf.name_scope('first_conv_layer_64_filter') as scope:
        layer_conv1, weights_conv1 = cnn.create_conv_layer(data, image_depth, filter_size1, num_filters1, name='1_conv_layer')
        layer_conv2, weights_conv2 = cnn.create_conv_layer(layer_conv1, num_filters1, filter_size2, num_filters2, name='2_conv_layer')
        layer_conv2_pool = cnn.pooling(layer_conv2, name='layer_2_pooling')

    with tf.variable_scope('secound_conv_layer_128_filter') as scope:
        layer_conv3, weights_conv3 = cnn.create_conv_layer(layer_conv2_pool, num_filters2, filter_size3, num_filters3, name='3_conv_layer')
        layer_conv4, weights_conv4 = cnn.create_conv_layer(layer_conv3, num_filters3, filter_size4, num_filters4, name='4_conv_layer')
        layer_conv4_pool = cnn.pooling(layer_conv4, name='layer_4_pooling')

    with tf.variable_scope('thirth_conv_layer_256_filter') as scope:
        layer_conv5, weights_conv5 = cnn.create_conv_layer(layer_conv4_pool, num_filters4, filter_size5, num_filters5, name='5_conv_layer')
        layer_conv6, weights_conv6 = cnn.create_conv_layer(layer_conv5, num_filters5, filter_size6, num_filters6, name='6_conv_layer')
        layer_conv6_pool = cnn.pooling(layer_conv6, name='layer_6_pooling')

    with tf.variable_scope('fourth_conv_layer_512_filter') as scope:
        layer_conv7, weights_conv7 = cnn.create_conv_layer(layer_conv6_pool, num_filters6, filter_size7, num_filters7, name='7_conv_layer')
        layer_conv8, weights_conv8 = cnn.create_conv_layer(layer_conv7, num_filters7, filter_size8, num_filters8, name='8_conv_layer')
        layer_conv8_pool = cnn.pooling(layer_conv8, name='layer_8_pooling')

    with tf.variable_scope('fivth_conv_layer_512_filter') as scope:
        layer_conv9, weights_conv9 = cnn.create_conv_layer(layer_conv8_pool, num_filters8, filter_size9, num_filters9, name='9_conv_layer')
        layer_conv10, weights_conv10 = cnn.create_conv_layer(layer_conv9, num_filters9, filter_size10, num_filters10, name='10_conv_layer')
        layer_conv10_pool = cnn.pooling(layer_conv10, name='layer_10_pooling')

    with tf.variable_scope('fully_connected_layer') as scope:
        #layer_flat, num_features = cnn.flatten_layer(layer_conv10_pool, name='flatten_layer')
        rnn_layer1 = cnn.create_RNN(layer_conv10_pool, rnn_size1, num_lable, keep_prob)




    return rnn_layer3
