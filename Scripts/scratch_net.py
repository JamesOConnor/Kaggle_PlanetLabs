import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback, EarlyStopping
from data_helper import DataHelper
from model_helper import ModelHelper

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import glob
import cv2
from tqdm import tqdm
import time


def process_predictions(predictions, labels, prediction_threshold):

	# TODO:
    print('Adjustments based on exclusive class plus probabilities of other classes')


if __name__ == '__main__':

    # Meep meep
    helper = DataHelper()
    model_helper = ModelHelper()

    # Get train and test filepaths
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    training = False
    load_prev = True

    # Training options
    num_epochs = 1
    batch_size = 24
    verbosity_level = 1
    image_size = (128,128) # inits tensor shapes as well as opencv resize in batch generators
    image_channels = 3
    output_size = 4

    # Image augmentation options
    image_augment_multiplier = 1 # goes over training set twice (with random flips and rotations) * this value
    horizontal_flip = True
    vertical_flip = True
    random_rotate = True

    # Other options
    prediction_threshold = 0.2 # for calling a spade a spade
    fun_loss = 'binary_crossentropy'    # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
    cool_optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)     # Adam optimizer with nesterov momentum

    # Model architecture definition
    with tf.device('/gpu:0'):
        weather_model = model_helper.get_weather_model(image_size=image_size, image_channels=image_channels, output_size=4)
        haze_model = model_helper.get_ground_model(image_size=image_size, image_channels=image_channels, output_size=13)
        clear_model = model_helper.get_ground_model(image_size=image_size, image_channels=image_channels, output_size=13)
        cloudy_model = model_helper.get_ground_model(image_size=image_size, image_channels=image_channels, output_size=13)
        pcloudy_model = model_helper.get_ground_model(image_size=image_size, image_channels=image_channels, output_size=13)

    # Loss function and GD algorithm
    weather_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    haze_model.compile(loss='binary_crossentropy', optimizer=cool_optimizer, metrics=['accuracy'])
    clear_model.compile(loss='binary_crossentropy', optimizer=cool_optimizer, metrics=['accuracy'])
    cloudy_model.compile(loss='binary_crossentropy', optimizer=cool_optimizer, metrics=['accuracy'])
    pcloudy_model.compile(loss='binary_crossentropy', optimizer=cool_optimizer, metrics=['accuracy'])

    # This is actually pretty cool
    # print(weather_model.summary())

    # TODO: early stopping will auto-stop training process if model stops learning after 3 epochs
    # earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

    # For logging
    # history = LossHistory()

    # X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validation_split_size) # TODO

    steps_per_epoch_train = len(train) / batch_size
    steps_per_epoch_train *= image_augment_multiplier

    # TODO: Use the history callback somehow
    # TODO: Poke around the early stopping parameters
    # TODO: Tensorboard

    # Load data
    print('Loading training data')
    train_x, train_y = helper.load_train_images(image_size)

    train_hx = helper.load_train_haralick()

    # Subset weather labels
    weather_labels = ['clear', 'cloudy', 'partly_cloudy', 'haze']
    ground_labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'conventional_mine',
                     'cultivation', 'habitation', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']

    weather_y = train_y[weather_labels]

    # Subset to secondary labels
    haze_y = train_y[train_y['haze'] == 1]
    clear_y = train_y[train_y['clear'] == 1]
    cloudy_y = train_y[train_y['cloudy'] == 1]
    pcloudy_y = train_y[train_y['partly_cloudy'] == 1]

    haze_y = train_y[train_y['haze'] == 1] # Selects rows that are haze
    haze_y = haze_y[ground_labels]         # Selects columns that are ground_labels

    clear_y = train_y[train_y['clear'] == 1]
    clear_y = clear_y[ground_labels]

    cloudy_y = train_y[train_y['cloudy'] == 1]
    cloudy_y = cloudy_y[ground_labels]

    pcloudy_y = train_y[train_y['partly_cloudy'] == 1]
    pcloudy_y = pcloudy_y[ground_labels]

    # Save model

    # Train model on batch data
    if training:

        weather_model.fit(train_x, weather_y.values, batch_size=batch_size, epochs=5, verbose=1)
        weather_model.save('weather_vgg.h5')

        haze_model.fit(train_x[train_y['haze'] == 1], haze_y.values, batch_size=batch_size, epochs=5, verbose=1)
        haze_model.save('haze_vgg.h5')

        clear_model.fit(train_x[train_y['clear'] == 1], clear_y.values, batch_size=batch_size, epochs=5, verbose=1)
        clear_model.save('clear_vgg.h5')

        cloudy_model.fit(train_x[train_y['cloudy'] == 1], cloudy_y.values, batch_size=batch_size, epochs=5, verbose=1)
        cloudy_model.save('cloudy_vgg.h5')

        pcloudy_model.fit(train_x[train_y['partly_cloudy'] == 1], pcloudy_y.values, batch_size=batch_size, epochs=5, verbose=1)
        pcloudy_model.save('pcloudy_vgg.h5')

    # Alternatively load
    if load_prev:
        weather_model = load_model('weather_vgg.h5')
        haze_model = load_model('haze_vgg.h5')
        clear_model = load_model('clear_vgg.h5')
        cloudy_model = load_model('cloudy_vgg.h5')
        pcloudy_model = load_model('pcloudy_vgg.h5')

    # In case some memory is required
    del train_x

    # Load test images
    test_x = helper.load_test_images(image_size)

    batch_size_test = 512 # use 523 or 12 as nicest integer divisors of len(test)
    steps_per_epoch_test = (len(test) / batch_size_test) + 1

    # Init the prediction dataframe
    predictions = pd.DataFrame(None, index=np.arange(len(test_x)), columns=train_y.columns)

    # Predict weather label
    weather_prob = weather_model.predict(test_x, batch_size=batch_size_test, verbose=1) # prob as output is softmax

    # Convert weather probabilities to selected class using argmax
    weather_pred = np.zeros_like(weather_prob)
    weather_pred[np.arange(len(weather_prob)), weather_prob.argmax(axis=1)] = 1

    # Assign weather predictions
    predictions.loc[:, weather_labels] = weather_pred

    # For any weather label W, call W_model on test data where W == 1, then assign to predictions data frame
    if any(predictions.haze == 1):
        haze_gr_pred = haze_model.predict(test_x[predictions.haze == 1], batch_size=batch_size_test, verbose=1)
        predictions.loc[predictions.haze == 1, ground_labels] = np.where(haze_gr_pred > prediction_threshold, 1, 0) # converts sigmoid outputs to one-hot on the fly

    if any(predictions.clear == 1):
        clear_gr_pred = clear_model.predict(test_x[predictions.clear == 1], batch_size=batch_size_test, verbose=1)
        predictions.loc[predictions.clear == 1, ground_labels] = np.where(clear_gr_pred > prediction_threshold, 1, 0)

    if any(predictions.cloudy == 1):
        cloudy_gr_pred = cloudy_model.predict(test_x[predictions.cloudy == 1], batch_size=batch_size_test, verbose=1)
        predictions.loc[predictions.cloudy == 1, ground_labels] = np.where(cloudy_gr_pred > prediction_threshold, 1, 0)

    if any(predictions.partly_cloudy == 1):
        pcloudy_gr_pred = pcloudy_model.predict(test_x[predictions.partly_cloudy == 1], batch_size=batch_size_test, verbose=1)
        predictions.loc[predictions.partly_cloudy == 1, ground_labels] = np.where(pcloudy_gr_pred > prediction_threshold, 1, 0)

    prediction_classes = np.zeros_like(predictions)
    prediction_classes[ np.where( predictions > prediction_threshold) ] = 1

    tag_info = pd.DataFrame([helper.get_label_matrix().mean(), predictions.mean()], index=['train_ratio', 'test_ratio'])
    print('Tags broken down by training and test set prediction ratio:')
    print(tag_info.transpose())

    submission_filename = 'current_submission.csv'
    print('Writing submission file: %s' % submission_filename)
    submission_df = helper.predictions_to_submissions(predictions.values, predictions.columns, test)
    submission_df.to_csv(submission_filename, header=True, index=False)


