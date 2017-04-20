import cv2
import numpy as np
import glob
import sklearn.linear_model as lm
import sklearn.ensemble as es

train = np.loadtxt('train.csv', delimiter=',', dtype=str)
classes = np.unique(train[1:,1])
class_ids = np.arange(classes.shape[0])

train_mapped_to_class_ids = np.zeros_like(train[1:,0])

mean_green = np.zeros_like(train[1:,0])
mean_red = np.zeros_like(train[1:,0])
mean_blue = np.zeros_like(train[1:,0])
std_green = np.zeros_like(train[1:,0])
std_red = np.zeros_like(train[1:,0])
std_blue = np.zeros_like(train[1:,0])

train_labels = train[1:,1]

for i in range(len(train[1:])):
    im = cv2.imread('train-jpg/train_%s.jpg'%(int(i)))
    mean_green[i] = float(im[:, 1].mean())
    mean_red[i] = float(im[:, 0].mean())
    mean_blue[i] = float(im[:, 2].mean())
    std_green[i] = float(im[:, 1].std())
    std_red[i] = float(im[:, 0].std())
    std_blue[i] = float(im[:, 2].std())

for n,i in enumerate(train_labels):
    ind = int(np.where(classes==i)[0])
    train_mapped_to_class_ids[n] = ind
    print n

mean_all = np.r_['1,2,0', mean_green, mean_red, mean_blue, std_green, std_red, std_blue].astype(float)
mean_green = mean_green.reshape(1,-1).T

rf = es.RandomForestClassifier()

rf.fit(mean_all, train_mapped_to_class_ids)

fns = glob.glob('test-jpg/*.jpg')

mean_green_test = np.zeros((len(fns)))
mean_red_test = np.zeros((len(fns)))
mean_blue_test = np.zeros((len(fns)))
std_green_test = np.zeros((len(fns)))
std_red_test = np.zeros((len(fns)))
std_blue_test = np.zeros((len(fns)))

for i,fn in enumerate(fns):
    print i
    im = cv2.imread(fn)
    mean_green_test[i] = float(im[:, 1].mean())
    mean_red_test[i] = float(im[:, 0].mean())
    mean_blue_test[i] = float(im[:, 2].mean())
    std_green_test[i] = float(im[:, 1].std())
    std_red_test[i] = float(im[:, 0].std())
    std_blue_test[i] = float(im[:, 2].std())

mean_all_test = np.r_['1,2,0', mean_green_test, mean_red_test, mean_blue_test, std_green_test, std_red_test, std_blue_test].astype(float)
output_classes = rf.predict(mean_all_test)

out_formatted = []
out_formatted.append(['image_name', 'tags'])

for n,i in enumerate(fns):
    print n
    out_formatted.append([i.split('\\')[1].split('.')[0], classes[output_classes[n]]])

np.savetxt('for_upload.csv', out_formatted, delimiter=',', fmt='%s')
