import pandas as pd
import numpy as np
import sklearn.ensemble as es
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import time
import pickle


def run_pretrained_model(config):

    # Read config parameters
    label_filepath = config['label_filepath']
    test_filepath = config['test_filepath']
    output_filepath = config['output_filepath']
    UID = config['pre_trained_UID']  # NOT generating a new UID
    model_filepath = config['model_directory'] + UID + '.model'

    # Read data
    print('Reading datasets')
    labels = pd.read_csv(label_filepath)
    test_data = pd.read_csv(test_filepath)

    # Map class ids
    classes = np.unique(labels.tags)
    train_ids = np.zeros(len(labels.tags))

    for n, i in enumerate(labels.tags):
        ind = int(np.where(classes == i)[0])
        train_ids[n] = ind

    # Train model
    model = pickle.load(open(model_filepath, 'rb'))
    print('Loaded pre-trained model: ', type(model))

    # Test model
    print('Evaluating on test data')
    pred = model.predict(test_data)

    # Write output
    print('Writing output: ', output_filepath)
    write_submission_file(pred, output_filepath, classes)


def run_new_model(config):

    # Read config parameters, generate UID
    label_filepath = config['label_filepath']
    train_filepath = config['train_filepath']
    test_filepath = config['test_filepath']
    UID = time.strftime("%Y%m%d-%H%M", time.gmtime()) # generate UID using formatted time
    model_filepath = config['model_directory'] + UID + '.model'
    config_filepath = config['model_directory'] + UID + '.p'
    output_filepath = config['output_directory'] + UID + '.csv'
    model = config['model']

    # Read data
    print('Reading datasets')
    labels = pd.read_csv(label_filepath)
    train_data = pd.read_csv(train_filepath)
    test_data = pd.read_csv(test_filepath)

    # Map class ids
    classes, train_ids = map_classes_to_id(labels.tags)

    # Train model
    print('Training model: ', type(model))
    start = time.time()
    model.fit(train_data, train_ids)
    end = time.time()
    del train_data 
    print('Trained model in: %fs' % (end - start))

    # Save model
    print('Saving model: ', model_filepath)
    pickle.dump(model, open(model_filepath, 'wb'))

    # Test model
    print('Testing model')
    pred = model.predict(test_data)

    # Write output
    print('Writing output: ', output_filepath)
    write_submission_file(pred, output_filepath, classes)

    # pickle.dump(config, open(config_filepath, 'wb'))


def write_submission_file(predictions, output_filepath, classes):

    output = pd.DataFrame(data=np.ndarray(shape=(len(predictions), 2)), columns=['image_name', 'tags'])

    for index, val in enumerate(predictions):
        output.loc[index] = ['test_%g' % int(index), classes[int(val)]]

    output.to_csv(output_filepath, index=False)

def map_classes_to_label(tags):
    """
    Map  
    """

    # map tags to integers
    # create list of list of mapped integers
    # get one hot encoder of list of list

    tagsL = [x for x in tags for x in x.split()]
    unique_tags = np.unique(tagsL)

    tagsList = [tag.split() for tag in tags]

    # Get tag_map dict
    tag_map = {}
    for index, tag in enumerate(unique_tags):
        tag_map[tag] = index

    # Map tagsList to intList
    intList = []
    # for index, row in enumerate(tagsList):
    #     for tag in row:
    #         # intList[]

    # pd.get_dummies('...')

    # for index, tag in enumerate(unique_tags):



    mlb = MultiLabelBinarizer(tagsL)


def map_classes_to_id(tags):

    # Map class ids
    classes = np.unique(tags)
    train_ids = np.zeros(len(tags))

    for index, tag in enumerate(tags):
        class_integer = int(np.where(classes == tag)[0])
        train_ids[index] = class_integer

    return classes, train_ids


if __name__ == '__main__':

    # Basic run configuration
    config = {'label_filepath' : '../../train.csv',
              'train_filepath' : '../data/train_colourharalick.csv',
              'test_filepath' : '../data/test_colourharalick.csv',
              'output_directory' : '../submissions/',
              'model_directory' : '../models/',
              'pre_trained_UID': '00000000-0000',   # only used in pre_trained_model(config)
               'model':OneVsRestClassifier(es.RandomForestClassifier(n_estimators=32, criterion="entropy", oob_score=False, n_jobs=-1, max_depth=12 )),
              }

    run_new_model(config)

    # run_pretrained_model(config)

    print('Done!')