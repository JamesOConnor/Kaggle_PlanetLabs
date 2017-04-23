import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')

def load_atm_labels(train):
    '''
    Separates atmospheric labels (where a minimum of 1 is required) from the land specific tags
    :param train: training csv read by pandas (pd.read_csv)
    :return: Atmospheric conditions in multiclass [0] and one hot[1] formats. Also return label mapping dictionary for multiclass and array of labels stripped of atmospheric tags
    '''
    classes = np.unique(train.values[:, 1])
    train_mapped_to_class_ids = np.zeros_like(train.values[:,1])
    train_labels = train.values[:, 1]
    for n,i in enumerate(train_labels):
        ind = int(np.where(classes==i)[0])
        train_mapped_to_class_ids[n] = ind
    train_mapped_to_class_ids = train_mapped_to_class_ids.astype(int)

    hazy, clear, cloudy, partly_cloudy = [], [], [], []
    atm_dict = {}
    atm_dict['0'] = 'hazy'
    atm_dict['1'] = 'clear'
    atm_dict['2'] = 'cloudy'
    atm_dict['3'] = 'partly_cloudy'

    labels_wo_atm = []
    train_labels = train.values[:,1]

    for n, i in enumerate(train_labels):
        if 'haze' in i:
            li = train_labels[n].split(' ')
            li.remove('haze')
            labels_wo_atm.append(' '.join(li))
            hazy.append(n)
        elif 'clear' in i:
            li = train_labels[n].split(' ')
            li.remove('clear')
            labels_wo_atm.append(' '.join(li))
            clear.append(n)
        elif 'partly_cloudy' in i:
            li = train_labels[n].split(' ')
            li.remove('partly_cloudy')
            labels_wo_atm.append(' '.join(li))
            cloudy.append(n)
        elif 'cloudy' in i:
            li = train_labels[n].split(' ')
            li.remove('cloudy')
            labels_wo_atm.append(' '.join(li))
            partly_cloudy.append(n)
    labels_wo_atm = np.array(labels_wo_atm)
    atm_cond = [hazy, clear, cloudy, partly_cloudy]
    atm = np.zeros_like(train_mapped_to_class_ids)
    y_train = []
    for n,i in enumerate(atm_cond):
        atm[np.array(i).astype(int)] = n
        prep = np.zeros_like(train_mapped_to_class_ids)
        prep[i] = 1
        y_train.append(prep)
    return atm, y_train, atm_dict, labels_wo_atm