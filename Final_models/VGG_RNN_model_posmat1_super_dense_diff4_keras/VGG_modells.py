import numpy as np
import tensorflow as tf
import CNN_utils as cnn
import keras

def VGG_A(keep_prob, split_size):
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
    rnn_size1 = 256
    rnn_size2 = 256
    fc_size1 = 256


    image_width = 300
    image_height = 300
    image_depth = 6
    num_lable = 8
    batch_size = 32

    imput_shape = [7, image_width, image_height, image_depth]

    model = keras.models.Sequential()

    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters1, (filter_size1, filter_size1), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters2, (filter_size2, filter_size2), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters3, (filter_size3, filter_size3), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters4, (filter_size4, filter_size4), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters3, (filter_size3, filter_size3), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters4, (filter_size4, filter_size4), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters5, (filter_size5, filter_size5), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters6, (filter_size6, filter_size6), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters7, (filter_size7, filter_size7), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters8, (filter_size8, filter_size8), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters9, (filter_size9, filter_size9), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters10, (filter_size10, filter_size10), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.Flatten()))

    model.add(keras.layers.recurrent.LSTM(256, return_sequences=True, dropout=keep_prob))
    model.add(keras.layers.recurrent.LSTM(256, return_sequences=True, dropout=keep_prob))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.Dense(fc_size1, activation='relu')))
    model.add(keras.layers.Dropout(keep_prob))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.Dense(num_lable, activation='softmax')))
    print(model.summary())

    return model
