import numpy as np
import tensorflow as tf
import CNN_utils as cnn

def VGG_B(keep_prob, data):
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
    fc_size1      = 4096
    fc_size2      = 5120
    fc_size3      = 1000

    image_width = 300
    image_height = 300
    image_depth = 6
    num_lable = 8

    keep_probability = 0.5




    with tf.name_scope('first_conv_layer_64_filter') as scope:
        layer_conv1, weights_conv1 = cnn.create_conv_layer(data, image_depth, filter_size1, num_filters1, name='1_conv_layer')
        layer_conv1_pool = cnn.pooling(layer_conv1, name='layer_1_pooling')

    with tf.variable_scope('secound_conv_layer_128_filter') as scope:
        layer_conv3, weights_conv3 = cnn.create_conv_layer(layer_conv1_pool, num_filters1, filter_size3, num_filters3, name='3_conv_layer')
        layer_conv3_pool = cnn.pooling(layer_conv3, name='layer_3_pooling')

    with tf.variable_scope('thirth_conv_layer_256_filter') as scope:
        layer_conv5, weights_conv5 = cnn.create_conv_layer(layer_conv3_pool, num_filters3, filter_size5, num_filters5, name='5_conv_layer')
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
        layer_flat, num_features = cnn.flatten_layer(layer_conv10_pool, name='flatten_layer')
        fc_layer1 = cnn.create_fully_connected_layer(layer_flat, num_features, fc_size1, name='1_fully_connected')
        fc_layer1_dropout = cnn.dropout(fc_layer1, keep_prob, name='1_layer_dropout')
        fc_layer2 = cnn.create_fully_connected_layer(fc_layer1_dropout, fc_size1, fc_size2, name='2_fully_connected')
        fc_layer2_dropout = cnn.dropout(fc_layer2, keep_prob, name='2_layer_dropout')
        fc_layer3 = cnn.create_fully_connected_layer(fc_layer2_dropout, fc_size2, fc_size3, name='3_fully_connected')
        fc_layer4 = cnn.create_fully_connected_layer(fc_layer3, fc_size3, num_lable, relu=False, name='4_fully_connected')

    return fc_layer4

def VGG_A(keep_prob, data):
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
    fc_size1      = 4096
    fc_size2      = 8192
    fc_size3      = 1000

    image_width = 300
    image_height = 300
    image_depth = 6
    num_lable = 8

    keep_probability = 0.5




    with tf.name_scope('first_conv_layer_64_filter') as scope:
        layer_conv1, weights_conv1 = cnn.create_conv_layer(data, image_depth, filter_size1, num_filters1, name='1_conv_layer')
        layer_conv1_pool = cnn.pooling(layer_conv1, name='layer_1_pooling')

    with tf.variable_scope('secound_conv_layer_128_filter') as scope:
        layer_conv3, weights_conv3 = cnn.create_conv_layer(layer_conv1_pool, num_filters1, filter_size3, num_filters3, name='3_conv_layer')
        layer_conv3_pool = cnn.pooling(layer_conv3, name='layer_3_pooling')

    with tf.variable_scope('thirth_conv_layer_256_filter') as scope:
        layer_conv5, weights_conv5 = cnn.create_conv_layer(layer_conv3_pool, num_filters3, filter_size5, num_filters5, name='5_conv_layer')
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
        layer_flat, num_features = cnn.flatten_layer(layer_conv10_pool, name='flatten_layer')
        fc_layer1 = cnn.create_fully_connected_layer(layer_flat, num_features, fc_size1, name='1_fully_connected')
        fc_layer1_dropout = cnn.dropout(fc_layer1, keep_prob, name='1_layer_dropout')
        fc_layer2 = cnn.create_fully_connected_layer(fc_layer1_dropout, fc_size1, fc_size2, name='2_fully_connected')
        fc_layer2_dropout = cnn.dropout(fc_layer2, keep_prob, name='2_layer_dropout')
        fc_layer3 = cnn.create_fully_connected_layer(fc_layer2_dropout, fc_size2, fc_size3, name='3_fully_connected')
        fc_layer4 = cnn.create_fully_connected_layer(fc_layer3, fc_size3, num_lable, relu=False, name='4_fully_connected')

    return fc_layer4

def VGG_C(keep_prob, data):
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
    filter_size7 = 3
    num_filters7 = 256
    #maxpool
    filter_size8 = 3
    num_filters8 = 512
    filter_size9 = 3
    num_filters9 = 512
    filter_size10 = 3
    num_filters10 = 512
    #maxpool
    filter_size11 = 3
    num_filters11 = 512
    filter_size12 = 3
    num_filters12 = 512
    filter_size13 = 3
    num_filters13 = 512
    #maxpool
    fc_size1      = 4096
    fc_size2      = 4096
    fc_size3      = 4096
    fc_size4      = 1000

    image_width = 300
    image_height = 300
    image_depth = 6
    num_lable = 8

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
        layer_conv7, weights_conv7 = cnn.create_conv_layer(layer_conv6, num_filters6, filter_size7, num_filters7, name='7_conv_layer')
        layer_conv7_pool = cnn.pooling(layer_conv7, name='layer_7_pooling')

    with tf.variable_scope('fourth_conv_layer_512_filter') as scope:
        layer_conv8, weights_conv8 = cnn.create_conv_layer(layer_conv7_pool, num_filters7, filter_size8, num_filters8, name='8_conv_layer')
        layer_conv9, weights_conv9 = cnn.create_conv_layer(layer_conv8, num_filters8, filter_size9, num_filters9, name='9_conv_layer')
        layer_conv10, weights_conv10 = cnn.create_conv_layer(layer_conv9, num_filters9, filter_size10, num_filters10, name='10_conv_layer')
        layer_conv10_pool = cnn.pooling(layer_conv10, name='layer_10_pooling')

    with tf.variable_scope('fivth_conv_layer_512_filter') as scope:
        layer_conv11, weights_conv11 = cnn.create_conv_layer(layer_conv10_pool, num_filters10, filter_size11, num_filters11, name='11_conv_layer')
        layer_conv12, weights_conv12 = cnn.create_conv_layer(layer_conv11, num_filters11, filter_size12, num_filters12, name='12_conv_layer')
        layer_conv13, weights_conv13 = cnn.create_conv_layer(layer_conv12, num_filters12, filter_size13, num_filters13, name='13_conv_layer')
        layer_conv13_pool = cnn.pooling(layer_conv13, name='layer_13_pooling')

    with tf.variable_scope('fully_connected_layer') as scope:
        layer_flat, num_features = cnn.flatten_layer(layer_conv13_pool, name='flatten_layer')
        fc_layer1 = cnn.create_fully_connected_layer(layer_flat, num_features, fc_size1, name='1_fully_connected')
        fc_layer1_dropout = cnn.dropout(fc_layer1, keep_prob, name='1_layer_dropout')
        fc_layer2 = cnn.create_fully_connected_layer(fc_layer1_dropout, fc_size1, fc_size2, name='2_fully_connected')
        fc_layer2_dropout = cnn.dropout(fc_layer2, keep_prob, name='2_layer_dropout')
        fc_layer3 = cnn.create_fully_connected_layer(fc_layer2_dropout, fc_size2, fc_size3, name='3_fully_connected')
        fc_layer3_dropout = cnn.dropout(fc_layer3, keep_prob, name='3_layer_dropout')
        fc_layer4 = cnn.create_fully_connected_layer(fc_layer3_dropout, fc_size3, fc_size4, name='4_fully_connected')
        fc_layer5 = cnn.create_fully_connected_layer(fc_layer4, fc_size4, num_lable, relu=False, name='5_fully_connected')

    return fc_layer5

def VGG_D(keep_prob, data):
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
    filter_size7 = 3
    num_filters7 = 256
    #maxpool
    filter_size8 = 3
    num_filters8 = 512
    filter_size9 = 3
    num_filters9 = 512
    filter_size10 = 3
    num_filters10 = 512
    #maxpool
    filter_size11 = 3
    num_filters11 = 512
    filter_size12 = 3
    num_filters12 = 512
    filter_size13 = 3
    num_filters13 = 512
    #maxpool
    fc_size1      = 2048
    fc_size2      = 4096
    fc_size3      = 1000

    image_width = 300
    image_height = 300
    image_depth = 6
    num_lable = 8

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
        layer_conv7, weights_conv7 = cnn.create_conv_layer(layer_conv6, num_filters6, filter_size7, num_filters7, name='7_conv_layer')
        layer_conv7_pool = cnn.pooling(layer_conv7, name='layer_7_pooling')

    with tf.variable_scope('fourth_conv_layer_512_filter') as scope:
        layer_conv8, weights_conv8 = cnn.create_conv_layer(layer_conv7_pool, num_filters7, filter_size8, num_filters8, name='8_conv_layer')
        layer_conv9, weights_conv9 = cnn.create_conv_layer(layer_conv8, num_filters8, filter_size9, num_filters9, name='9_conv_layer')
        layer_conv10, weights_conv10 = cnn.create_conv_layer(layer_conv9, num_filters9, filter_size10, num_filters10, name='10_conv_layer')
        layer_conv10_pool = cnn.pooling(layer_conv10, name='layer_10_pooling')

    with tf.variable_scope('fivth_conv_layer_512_filter') as scope:
        layer_conv11, weights_conv11 = cnn.create_conv_layer(layer_conv10_pool, num_filters10, filter_size11, num_filters11, name='11_conv_layer')
        layer_conv12, weights_conv12 = cnn.create_conv_layer(layer_conv11, num_filters11, filter_size12, num_filters12, name='12_conv_layer')
        layer_conv13, weights_conv13 = cnn.create_conv_layer(layer_conv12, num_filters12, filter_size13, num_filters13, name='13_conv_layer')
        layer_conv13_pool = cnn.pooling(layer_conv13, name='layer_13_pooling')

    with tf.variable_scope('fully_connected_layer') as scope:
        layer_flat, num_features = cnn.flatten_layer(layer_conv13_pool, name='flatten_layer')
        fc_layer1 = cnn.create_fully_connected_layer(layer_flat, num_features, fc_size1, name='1_fully_connected')
        fc_layer1_dropout = cnn.dropout(fc_layer1, keep_prob, name='1_layer_dropout')
        fc_layer2 = cnn.create_fully_connected_layer(fc_layer1_dropout, fc_size1, fc_size2, name='2_fully_connected')
        fc_layer2_dropout = cnn.dropout(fc_layer2, keep_prob, name='2_layer_dropout')
        fc_layer3 = cnn.create_fully_connected_layer(fc_layer2_dropout, fc_size2, fc_size3, name='3_fully_connected')
        fc_layer5 = cnn.create_fully_connected_layer(fc_layer3, fc_size3, num_lable, relu=False, name='5_fully_connected')

    return fc_layer5

def VGG_E(keep_prob, data):
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
    filter_size7 = 3
    num_filters7 = 256
    #maxpool
    filter_size8 = 3
    num_filters8 = 512
    filter_size9 = 3
    num_filters9 = 512
    filter_size10 = 3
    num_filters10 = 512
    #maxpool
    filter_size11 = 3
    num_filters11 = 512
    filter_size12 = 3
    num_filters12 = 512
    filter_size13 = 3
    num_filters13 = 512
    #maxpool
    fc_size1      = 2048
    fc_size2      = 2048
    fc_size3      = 1000

    image_width = 300
    image_height = 300
    image_depth = 6
    num_lable = 8

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
        layer_conv7, weights_conv7 = cnn.create_conv_layer(layer_conv6, num_filters6, filter_size7, num_filters7, name='7_conv_layer')
        layer_conv7_pool = cnn.pooling(layer_conv7, name='layer_7_pooling')

    with tf.variable_scope('fourth_conv_layer_512_filter') as scope:
        layer_conv8, weights_conv8 = cnn.create_conv_layer(layer_conv7_pool, num_filters7, filter_size8, num_filters8, name='8_conv_layer')
        layer_conv9, weights_conv9 = cnn.create_conv_layer(layer_conv8, num_filters8, filter_size9, num_filters9, name='9_conv_layer')
        layer_conv10, weights_conv10 = cnn.create_conv_layer(layer_conv9, num_filters9, filter_size10, num_filters10, name='10_conv_layer')
        layer_conv10_pool = cnn.pooling(layer_conv10, name='layer_10_pooling')

    with tf.variable_scope('fivth_conv_layer_512_filter') as scope:
        layer_conv11, weights_conv11 = cnn.create_conv_layer(layer_conv10_pool, num_filters10, filter_size11, num_filters11, name='11_conv_layer')
        layer_conv12, weights_conv12 = cnn.create_conv_layer(layer_conv11, num_filters11, filter_size12, num_filters12, name='12_conv_layer')
        layer_conv13, weights_conv13 = cnn.create_conv_layer(layer_conv12, num_filters12, filter_size13, num_filters13, name='13_conv_layer')
        layer_conv13_pool = cnn.pooling(layer_conv13, name='layer_13_pooling')

    with tf.variable_scope('fully_connected_layer') as scope:
        layer_flat, num_features = cnn.flatten_layer(layer_conv13_pool, name='flatten_layer')
        fc_layer1 = cnn.create_fully_connected_layer(layer_flat, num_features, fc_size1, name='1_fully_connected')
        fc_layer1_dropout = cnn.dropout(fc_layer1, keep_prob, name='1_layer_dropout')
        fc_layer2 = cnn.create_fully_connected_layer(fc_layer1_dropout, fc_size1, fc_size2, name='2_fully_connected')
        fc_layer2_dropout = cnn.dropout(fc_layer2, keep_prob, name='2_layer_dropout')
        fc_layer3 = cnn.create_fully_connected_layer(fc_layer2_dropout, fc_size2, fc_size3, name='3_fully_connected')
        fc_layer5 = cnn.create_fully_connected_layer(fc_layer3, fc_size3, num_lable, relu=False, name='5_fully_connected')

    return fc_layer5

def VGG_F(keep_prob, data):
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
    fc_size1      = 8192
    fc_size2      = 16384
    fc_size3      = 1000

    image_width = 300
    image_height = 300
    image_depth = 6
    num_lable = 8

    keep_probability = 0.5




    with tf.name_scope('first_conv_layer_64_filter') as scope:
        layer_conv1, weights_conv1 = cnn.create_conv_layer(data, image_depth, filter_size1, num_filters1, name='1_conv_layer')
        layer_conv1_pool = cnn.pooling(layer_conv1, name='layer_1_pooling')

    with tf.variable_scope('secound_conv_layer_128_filter') as scope:
        layer_conv3, weights_conv3 = cnn.create_conv_layer(layer_conv1_pool, num_filters1, filter_size3, num_filters3, name='3_conv_layer')
        layer_conv3_pool = cnn.pooling(layer_conv3, name='layer_3_pooling')

    with tf.variable_scope('thirth_conv_layer_256_filter') as scope:
        layer_conv5, weights_conv5 = cnn.create_conv_layer(layer_conv3_pool, num_filters3, filter_size5, num_filters5, name='5_conv_layer')
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
        layer_flat, num_features = cnn.flatten_layer(layer_conv10_pool, name='flatten_layer')
        fc_layer1 = cnn.create_fully_connected_layer(layer_flat, num_features, fc_size1, name='1_fully_connected')
        fc_layer1_dropout = cnn.dropout(fc_layer1, keep_prob, name='1_layer_dropout')
        fc_layer2 = cnn.create_fully_connected_layer(fc_layer1_dropout, fc_size1, fc_size2, name='2_fully_connected')
        fc_layer2_dropout = cnn.dropout(fc_layer2, keep_prob, name='2_layer_dropout')
        fc_layer3 = cnn.create_fully_connected_layer(fc_layer2_dropout, fc_size2, fc_size3, name='3_fully_connected')
        fc_layer4 = cnn.create_fully_connected_layer(fc_layer3, fc_size3, num_lable, relu=False, name='4_fully_connected')

    return fc_layer4
