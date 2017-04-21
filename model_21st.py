import cv2
import numpy as np
import glob
import sklearn.ensemble as es

train = np.loadtxt('train.csv', delimiter=',', dtype=str)
classes = np.unique(train[1:,1])
class_ids = np.arange(classes.shape[0])

train_mapped_to_class_ids = np.zeros_like(train[1:,0])
train_labels = train[1:,1]
for n,i in enumerate(train_labels):
    ind = int(np.where(classes==i)[0])
    train_mapped_to_class_ids[n] = ind
    print n
train_mapped_to_class_ids = train_mapped_to_class_ids.astype(int)

current_run = True

xvec = np.zeros((len(train_labels), 21))

if current_run == False:

    orb_det = cv2.ORB()
    sift_det = cv2.SIFT()
    surf_det = cv2.SURF()

    for i in range(len(train[1:])):
        print i
        im = cv2.imread('train-jpg/train_%s.jpg'%(int(i)))
        for band in range(3):
            xvec[i,band] =  float(im[:, :, band].mean())
            xvec[i, band+3] = float(im[:, :, band].std())
            fft = abs(np.fft.fftshift(np.fft.fft2(im[:, :, band])))
            xvec[i, band + 6] = fft.mean()
            xvec[i, band + 9] = fft.std()
            xvec[i, band + 12] = len(orb_det.detect(im[:, :, band]))
            xvec[i, band + 15] = len(sift_det.detect(im[:, :, band]))
            xvec[i, band + 18] = len(surf_det.detect(im[:, :, band]))


    feats_all = np.r_['1,2,0', surf_feats_blue,surf_feats_red,surf_feats_green,orb_feats_blue,orb_feats_red,orb_feats_green, \
                 sift_feats_green, sift_feats_blue, sift_feats_red]

    mean_all = np.r_['1,2,0', mean_green, mean_red, mean_blue, std_green, std_red, std_blue, \
        mean_fft_green, mean_fft_red, mean_fft_blue, std_fft_green, std_fft_red, std_fft_blue, \
        feats_all].astype(float)
else:
    mean_all = np.loadtxt('xvecs/21_04_2017/xvec_means_stds_ffts_feats.csv', delimiter=',', dtype=float)

rf = es.RandomForestClassifier(n_jobs=2)
rf.fit(mean_all, train_mapped_to_class_ids)
