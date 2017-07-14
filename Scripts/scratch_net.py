import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback, EarlyStopping
from batch_generator import *

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import glob
import cv2
from tqdm import tqdm
import time

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def get_model(image_size=(32,32), image_channels=3):

    output_size = 17 # I guess this doesn't change that much

    model = Sequential()

    model.add(Conv2D(128,
                     input_shape=(*image_size, image_channels),
                     kernel_size=(5, 5),
                     kernel_initializer='random_uniform',
                     bias_initializer='zeros',
                     use_bias=True,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64,
                     kernel_size=(5, 5),
                     kernel_initializer='random_uniform',
                     bias_initializer='zeros',
                     use_bias=True,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     kernel_initializer='random_uniform',
                     bias_initializer='zeros',
                     use_bias=True,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_size, activation='sigmoid'))

    return model

def vgg16_model(image_size=(32,32), image_channels=3):

    output_size = 17
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(*image_size, image_channels)))
    #model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    #model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    #model.add( Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    #
    # # Block 5
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(256, activation='relu', name='fc1'))
    model.add(Dense(256, activation='relu', name='fc2'))
    model.add(Dense(output_size, activation='sigmoid', name='predictions'))
	
    return model


def get_amz_model(image_size=(32,32), image_channels=3):

    output_size = 17 # I guess this doesn't change that much

    model = Sequential()

    model.add(BatchNormalization(input_shape=(*image_size, image_channels)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='sigmoid'))

    return model


def predictions_to_submissions(predictions, labels, test):
    '''
    Converts outputs from CNNs to file formatted for submission to Kaggle
    :param oh_array: array with binary values representing tags
    :return: array to be written to file
    '''

    out_formatted = []

    for index, image_name in enumerate(test.image_name):

        tags_index = np.where(predictions[index]==1)[0]

        tags = ' '.join(labels[tags_index])

        out_formatted.append([image_name, tags])

    return pd.DataFrame(out_formatted, columns=['image_name', 'tags'])


def process_predictions(predictions, labels, prediction_threshold):

	# TODO: 


if __name__ == '__main__':

    # Get train and test filepaths
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Training options
    num_epochs = 1
    batch_size = 24
    verbosity_level = 1
    image_size = (128,128) # inits tensor shapes as well as opencv resize in batch generators
    image_channels = 3

    # Image augmentation options
    image_augment_multiplier = 1 # goes over training set twice (with random flips and rotations) * this value
    horizontal_flip = True
    vertical_flip = True
    random_rotate = True

    # Other options
    prediction_threshold = 0.2 # for calling a spade a spade
    fun_loss = 'binary_crossentropy'    # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
    cool_optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)     # Adam optimizer with nesterov momentum

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))
    labels = np.array(labels)

    # Model architecture definition
    with tf.device('/gpu:0'):
        model = vgg16_model(image_size=image_size, image_channels=image_channels)

    # Loss function and GD algorithm
    model.compile(loss=fun_loss,
                  optimizer=cool_optimizer,
                  metrics=['accuracy'])

    # This is actually pretty cool
    print(model.summary())

    # TODO: early stopping will auto-stop training process if model stops learning after 3 epochs
    earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

    # For logging
    history = LossHistory()

    # X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validation_split_size) # TODO

    #
    steps_per_epoch_train = len(train) / batch_size
    steps_per_epoch_train *= image_augment_multiplier

    # TODO: Use the history callback somehow
    # TODO: Poke around the early stopping parameters
    # TODO: Tensorboard

    # Train model on batch data
    model.fit_generator(train_batch_generator(batch_size, image_size),
                        steps_per_epoch_train,
                        epochs=num_epochs,
                        verbose=1,
                        # callbacks = [history, earlyStopping]
                        # validation_data=(X_valid, y_valid) # TODO using generator somehow .. (pre split the indices and save in class)
                        )

    # Save model
    model_path = 'vgg_3.h5'
    model.save(model_path)

    # Alternatively ..
    # model = load_model(model_path)

    # Test model

    batch_size_test = 64 # use 523 or 12 as nicest integer divisors of len(test)
    steps_per_epoch_test = (len(test) / batch_size_test) + 1
    predictions = model.predict_generator(test_batch_generator(batch_size_test, image_size), steps_per_epoch_test, verbose=1)
		
    # p_valid = model.predict(X_valid)
    # fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')
    # info = (history.train_losses, history.val_losses, fbeta_score)

    prediction_classes = np.zeros_like(predictions)
    prediction_classes[ np.where( predictions > prediction_threshold) ] = 1

    predictions_df = pd.DataFrame(prediction_classes, columns=labels)
    predictions_df.to_csv('predictions_onehot.csv', header=True)

    print('------------------------------------------')
    print('Percentage of images with tag in test set: ')
    print(predictions_df.mean())
    print('------------------------------------------')

    submission_filename = 'current_submission.csv'
    print('Writing submission file: %s' % submission_filename)
    submission_df = predictions_to_submissions(prediction_classes, labels, test)
    submission_df.to_csv(submission_filename, header=True, index=False)


