import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
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


if __name__ == '__main__':

    # Training options
    training = False
    restore_model = False
    testing = True
    num_epochs = 5
    batch_size = 128
    verbosity_level = 1
    image_size = (128,128) # inits tensor shapes as well as opencv resize in batch generators
    image_channels = 3
    aux_input_size = 66  # Number of features in auxiliary input (haralick + colour stats from L*a*b*)
    output_size = 17 # Can finetune the model at a later stage
    validation_epoch = 3  # Evaluates validation set every THIS epoch
    validation_split = 0.2 # percentage of training set to use as validation set
    checkpoint_filepath = '../models/functional_3NN_cp.hdf5'
    final_filepath = '../models/functional_3NN_final.hdf5'

    # Model options
    prediction_threshold = 0.2 # for calling a spade a spade
    fun_loss = 'binary_crossentropy'    # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
    cool_optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)     # Adam optimizer with nesterov momentum

    # Image augmentation options
    image_augment_multiplier = 1 # goes over training set twice (with random flips and rotations) * this value
    horizontal_flip = True
    vertical_flip = True
    random_rotate = True

    # Init helper classes
    helper = DataHelper()
    model_helper = ModelHelper()

    # Get train and test filepaths
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Load training labels (used as indices when creating batches)
    train_Y = helper.get_label_matrix()

    if restore_model:
        print('Restoring model: %s' % final_filepath)
        weather_model = load_model(final_filepath)
    else:
        # Model architecture definition
        with tf.device('/gpu:0'):
            weather_model = model_helper.get_functional_model(image_size, image_channels, aux_input_size, output_size)

        weather_model.compile(optimizer='Adam',
                              loss={'main_output': 'binary_crossentropy',
                                    'cnn_output': 'binary_crossentropy'},
                              loss_weights={'main_output': 1.,
                                            'cnn_output': .2}
                              )

    if training:
        # Split into training and validation set
        valid_Y = train_Y.sample(frac=validation_split, replace=False, random_state=1337) # TODO: Rename valid_set
        train_Y = train_Y.drop(valid_Y.index)
        best_metric = 0 # depends on what metric we're choosing

        for epoch in range(num_epochs):

            # Generate new batches each epoch
            train_gen = helper.batch_generator(list(train_Y.index), batch_size, shuffle=True, image_size=image_size, augment_image=True, columns_subset=None)
            valid_gen = helper.batch_generator(list(valid_Y.index), batch_size, shuffle=False, image_size=image_size, augment_image=False, columns_subset=None)

            epoch_loss = []
            batch_idx = 0
            # Training loop
            for train_X, train_AUX, train_Yi in train_gen:

                train_loss = weather_model.train_on_batch({'cnn_input': train_X, 'aux_input': train_AUX},
                                             {'cnn_output': train_Yi, 'main_output': train_Yi})

                # print('batch [%d] : loss %f \t cnn_loss %f \t main_output_loss %f' % (batch_idx, train_loss[0], train_loss[1], train_loss[2]))
                batch_idx += 1
                epoch_loss.append(train_loss)

            epoch_loss = np.mean(epoch_loss, axis=0) # reduce to average loss over
            print('EPOCH [%d] : loss %f \t cnn_loss %f \t main_output_loss %f' % (epoch, epoch_loss[0], epoch_loss[1], epoch_loss[2]))

            # Validation loop
            if epoch % validation_epoch == 0:

                predictions  = []
                actual = []
                for valid_X, valid_AUX, valid_Yi in valid_gen:

                    valid_pred_A, valid_pred_B = weather_model.predict_on_batch({'cnn_input': valid_X, 'aux_input': valid_AUX})

                    prediction_vector = helper.process_predictions(valid_pred_A, prediction_threshold)
                    predictions.append(prediction_vector)
                    actual.append(valid_Yi)

                predictions = np.vstack(predictions)
                actual = np.vstack(actual)

                summary_metrics = helper.calculate_metrics(predictions, actual)

                print('VALIDATION EPOCH %d\t Correct: %0.3f\t Accuracy: %0.3f \t Precision: %0.3f \t Sensitivity: %0.3f \t Specificity: %0.3f \t ' % (epoch, summary_metrics[0], summary_metrics[1], summary_metrics[2], summary_metrics[3], summary_metrics[4]))

                # Checkpoint model at best validation loss
                if summary_metrics[0] < best_metric:
                    best_metric = summary_metrics[0]

                    print('New best metric: %0.5f ' % best_metric)
                    print('Checkpointing model: %s ' % checkpoint_filepath)
                    weather_model.save(checkpoint_filepath)

            weather_model.save(final_filepath)

    if testing:

        # Restore model
        weather_model = load_model(final_filepath)

        # Init the prediction dataframe
        predictions = pd.DataFrame(None, columns=train_Y.columns)

        test_gen = helper.test_batch_generator(list(test.index), batch_size, shuffle=False, image_size=image_size,
                                               augment_image=False, columns_subset=None)

        predictions = []
        # Training loop
        for test_X, test_AUX in test_gen:

            test_pred_A, test_pred_B = weather_model.predict_on_batch({'cnn_input': train_X, 'aux_input': train_AUX})

            prediction_vector = helper.process_predictions(test_pred_A, prediction_threshold)

            predictions.append(prediction_vector)
            # actual.append(test_Yi)

        predictions = np.vstack(predictions)
        actual = np.vstack(actual)

        summary_metrics = helper.calculate_metrics(predictions, actual)

        print('TEST SET \t Correct: %0.3f\t Accuracy: %0.3f \t Precision: %0.3f \t Sensitivity: %0.3f \t Specificity: %0.3f \t ' % (
            epoch, summary_metrics[0], summary_metrics[1], summary_metrics[2], summary_metrics[3], summary_metrics[4]))


    prediction_classes = np.zeros_like(predictions)
    prediction_classes[ np.where( predictions > prediction_threshold) ] = 1

    tag_info = pd.DataFrame([helper.get_label_matrix().mean(), predictions.mean()], index=['train_ratio', 'test_ratio'])
    print('Tags broken down by training and test set prediction ratio:')
    print(tag_info.transpose())

    submission_filename = 'current_submission.csv'
    print('Writing submission file: %s' % submission_filename)
    df_predictions = pd.DataFrame(predictions, columns=train_Y.columns)
    submission_df = helper.predictions_to_submissions(df_predictions.values, df_predictions.columns, test)
    submission_df.to_csv(submission_filename, header=True, index=False)




    # # Subset to secondary labels
    # haze_y = train_Y[train_Y['haze'] == 1]
    # clear_y = train_Y[train_Y['clear'] == 1]
    # cloudy_y = train_Y[train_Y['cloudy'] == 1]
    # pcloudy_y = train_Y[train_Y['partly_cloudy'] == 1]
    #
    # haze_y = train_Y[train_Y['haze'] == 1] # Selects rows that are haze
    # haze_y = haze_y[helper.ground_labels]         # Selects columns that are ground_labels
    #
    # clear_y = train_Y[train_Y['clear'] == 1]
    # clear_y = clear_y[helper.ground_labels]
    #
    # cloudy_y = train_Y[train_Y['cloudy'] == 1]
    # cloudy_y = cloudy_y[helper.ground_labels]
    #
    # pcloudy_y = train_Y[train_Y['partly_cloudy'] == 1]
    # pcloudy_y = pcloudy_y[helper.ground_labels]
