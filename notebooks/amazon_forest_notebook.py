import sys
sys.path.append('../src')
sys.path.append('../tests')

import gc
import numpy as np
import pandas as pd
from itertools import chain

from data_helper import *
from keras_helper import AmazonKerasClassifier
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint


def build_classifier(output_layers):
    classifier = AmazonKerasClassifier()
    classifier.add_conv_layer(img_resize)
    classifier.add_flatten_layer()
    classifier.add_ann_layer(output_layers)
    return classifier

### Config ###

img_resize = (64, 64)
destination_path = "../input/"
train_data_folder = "train-jpg"
channel_mode = 1  # If channel mode == 2, 6 band images will be loaded TODO
validation_split_size = 0.2
batch_size = 128
test_data_folder = 'test-jpg'
additional_data_folder = 'test-jpg-additional'

### End config ###

train_csv_file = destination_path + 'train_v2.csv'
labels_df = pd.read_csv(train_csv_file)

labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)

x_train, y_train, y_map = preprocess_train_data('%s/%s/'%(destination_path, train_data_folder), train_csv_file, img_resize)


labels_df = pd.read_csv(train_csv_file)
labels = sorted(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
labels_map = {l: i for i, l in enumerate(labels)}
y_map = {v: k for k, v in labels_map.items()}

weather_labels = [labels_map[i] for i in ['clear', 'cloudy', 'haze', 'partly_cloudy']]
non_weather_labels = np.setdiff1d(range(17), weather_labels)
y_weather = y_train[:,np.array(weather_labels)]

weather_nn = build_classifier(4)


for weather in ['clear', 'cloudy', 'haze', 'partly_cloudy']:
    classifier = build_classifier(13)
    weather_ind = labels_map[weather]
    weather_subset = np.where(y_train[:,weather_ind] == 1)[0]
    x_current, y_current = x_train[weather_subset], y_train[weather_subset][:,non_weather_labels]
    if x_current.shape[0] < 5000:
        x_current, y_current = np.vstack((x_current, np.rot90(x_current,k = 1, axes=(1,2)), np.rot90(x_current, k = 2, axes=(1,2)),np.rot90(x_current, k = 3, axes=(1,2)))), np.vstack((y_current, y_current, y_current, y_current))
    filepath="%s_weights.best.hdf5"%weather
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    train_losses, val_losses = [], []
    epochs_arr = [10, 5, 5]
    learn_rates = [0.001, 0.0001, 0.00001]
    for learn_rate, epochs in zip(learn_rates, epochs_arr):
        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model(x_current, y_current, learn_rate, epochs,
                                                                              batch_size, validation_split_size=validation_split_size,
                                                                               train_callbacks=[checkpoint])
        print(fbeta_score)
        train_losses += tmp_train_losses
        val_losses += tmp_val_losses



# b = optimise_f2_thresholds(y_valid, classifier.predict(X_valid))
filepath = "weather_weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
train_losses, val_losses = [], []
epochs_arr = [15, 7, 7]
learn_rates = [0.001, 0.0001, 0.00001]
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    x_current = x_train[:, weather_labels]
    tmp_train_losses, tmp_val_losses, fbeta_score = weather_nn.train_model(x_current, y_train, learn_rate, epochs,
                                                                           batch_size,
                                                                           validation_split_size=validation_split_size,
                                                                           train_callbacks=[checkpoint])
    print(fbeta_score)
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses


# <markdowncell>

# ## Load Best Weights

# <markdowncell>

# Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training

# <codecell>

classifier.load_weights("weights.best.hdf5")
print("Weights loaded")

# <markdowncell>

# ## Monitor the results

# <markdowncell>

# Check that we do not overfit by plotting the losses of the train and validation sets

# <codecell>

# <markdowncell>

# Look at our fbeta_score

# <codecell>

# <markdowncell>

# Before launching our predictions lets preprocess the test data and delete the old training data matrices

# <codecell>


x_test, x_test_filename = data_helper.preprocess_test_data('../input/test_tif_reshape2/', img_resize)
# Predict the labels of our x_test images
predictions = classifier.predict(x_test)

# <markdowncell>

# Now lets launch the predictions on the additionnal dataset (updated on 05/05/2017 on Kaggle)

# <codecell>

del x_test
gc.collect()

x_test, x_test_filename_additional = data_helper.preprocess_test_data('../input/test_tif_additional2/', img_resize)
new_predictions = classifier.predict(x_test)

del x_test
gc.collect()
predictions = np.vstack((predictions, new_predictions))
x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              predictions[0]))

# <markdowncell>

# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class.
# 
# To do so we will take the median value of each classes.

# <codecell>

# For now we'll just put all thresholds to 0.2 
thresholds = optimise_f2_thresholds(y_valid, classifier.predict(X_valid))

# TODO complete


# <markdowncell>

# Now lets map our predictions to their tags and use the thresholds we just retrieved

# <codecell>

predicted_labels = classifier.map_predictions(predictions, y_map, thresholds)

# <markdowncell>

# Finally lets assemble and visualize our prediction for the test dataset

# <codecell>

tags_list = [None] * len(predicted_labels)
for i, tags in enumerate(predicted_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]

# <codecell>

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
final_df.head()

# <codecell>

# <markdowncell>

# If there is a lot of `primary` and `clear` tags, this final dataset may be legit...

# <markdowncell>

# And save it to a submission file

# <codecell>

final_df.to_csv('../submission_file.csv', index=False)
classifier.close()

# <markdowncell>

# That's it, we're done!
