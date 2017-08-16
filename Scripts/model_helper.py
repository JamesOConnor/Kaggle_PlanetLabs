import pandas as pd
import numpy as np
import cv2
import random
from tqdm import tqdm
from keras.datasets import mnist
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback, EarlyStopping
import glob


class ModelHelper():

    def __init__(self):
        pass

    def get_functional_model(self, image_size=(32,32), image_channels=3, aux_input_size=None, output_size=None):

        # CNN Input
        cnn_input = Input(shape=(*image_size, image_channels), name='cnn_input')

        # Block 1
        x = BatchNormalization(name="block1_batch_norm")(cnn_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name="block1_conv2d_1")(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name="block1_conv2d_2")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = BatchNormalization(name="block2_batch_norm")(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name="block2_conv2d_1")(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name="block2_conv2d_2")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = BatchNormalization(name="block3_batch_norm")(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name="block3_conv2d_1")(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name="block3_conv2d_2")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Dense layer
        x = Flatten()(x)
        x = Dense(2048, activation='relu')(x)
        cnn_last = Dense(2048, activation='relu')(x)

        # CNN Output
        cnn_output = Dense(output_size, activation='sigmoid', name='cnn_output')(cnn_last)

        # Auxiliary input
        aux_input = Input(shape=(aux_input_size,), name='aux_input')
        x = concatenate([aux_input, cnn_last])
        x = Dense(2048, activation='relu')(x)
        x = Dense(2048, activation='relu')(x)

        # Main output
        main_output = Dense(output_size, activation='sigmoid', name='main_output')(x)

        # Create the Model object
        model = Model(inputs=[cnn_input, aux_input], outputs=[cnn_output, main_output])

        return model

    def get_haralick_model(self, output_size, output_activation):

        input = Input(shape=(66,))
        dense1 = Dense(512,activation="sigmoid")(input)
        dense2 = Dense(512, activation="sigmoid")(dense1)

        model = Sequential()
        model.add(Dense(256, input_shape=(66,)), activation="sigmoid")
        model.add(Dense(256, activation="sigmoid"))
        model.add(Dense(output_size, activation=output_activation, name='predictions'))

    def get_weather_model(self, image_size=(32, 32), image_channels=3, output_size=4):

        model = Sequential()

        # model.add(BatchNormalization(input_shape=(*image_size, image_channels)))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',
                         input_shape=(*image_size, image_channels)))
        # model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        # model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add( Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        # model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(512, activation='relu', name='fc1'))
        # model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu', name='fc2'))
        # model.add(Dropout(0.25))
        model.add(Dense(output_size, activation='softmax', name='predictions'))

        return model

    def get_ground_model(self, image_size=(32, 32), image_channels=3, output_size=13):

        model = Sequential()

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',
                         input_shape=(*image_size, image_channels)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(256, activation='relu', name='fc1'))
        model.add(Dense(256, activation='relu', name='fc2'))
        model.add(Dense(output_size, activation='sigmoid', name='predictions'))

        return model



    def vgg16_model(image_size=(32, 32), image_channels=3, output_size=17):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',
                         input_shape=(*image_size, image_channels)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(256, activation='relu', name='fc1'))
        model.add(Dense(256, activation='relu', name='fc2'))
        model.add(Dense(output_size, activation='sigmoid', name='predictions'))

        return model