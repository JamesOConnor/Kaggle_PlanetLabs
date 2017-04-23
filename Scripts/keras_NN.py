import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
from tqdm import tqdm
import glob
from oh_to_labs import *

x_train = []
y_train = []

train = pd.read_csv('train.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(train.values, miniters=1000):
    img = cv2.imread('train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (64, 64)))
    y_train.append(targets)

y_train = np.array(y_train, np.uint8)  # ints for NN training set
x_train = np.array(x_train, np.float16) / 255.  # Convert to floats for CNN input

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(64, 64, 3)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=5,
          verbose=1)

fns = glob.glob('test-jpg/*.jpg')
x_test = []

for fn in tqdm(fns, miniters=1000):
    img = cv2.imread(fn)
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_test.append(cv2.resize(img, (64, 64)))

p_valid = model.predict(x_test, batch_size=128)
p_valid_classes = np.zeros_like(p_valid)
p_valid_classes[np.where(p_valid>.2)] = 1
for_sub = oh_to_labs(p_valid_classes)

