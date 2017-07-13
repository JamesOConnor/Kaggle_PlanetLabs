import numpy as np
import pandas as pd
import glob as glob

train = pd.read_csv('train.csv')
fns = glob.glob('test-jpg/*.jpg')
flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))
labels = np.array(labels)

def oh_to_labs(oh_array):
    '''
    Converts outputs from CNNs to file formatted for submission to Kaggle
    :param oh_array: array with binary values representing tags
    :return: array to be written to file
    '''

    out_formatted = []
    out_formatted.append(['image_name', 'tags'])
    for n,i in enumerate(fns):
        tags = np.where(oh_array[n]==1)[0]
        final_tags = ' '.join(labels[tags])
        out_formatted.append([i.split('\\')[1].split('.')[0], final_tags])
    return np.array(out_formatted)
