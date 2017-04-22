import cv2
import numpy as np
import mahotas as mh

def calc_basic_stats():
  '''
  Calculates and returns basic stats for all bands using numpy
  '''
  train = np.loadtxt('train.csv', delimiter=',', dtype=str)
  mean_green = np.zeros_like(train[1:, 0])
  mean_red = np.zeros_like(train[1:, 0])
  mean_blue = np.zeros_like(train[1:, 0])
  std_green = np.zeros_like(train[1:, 0])
  std_red = np.zeros_like(train[1:, 0])
  std_blue = np.zeros_like(train[1:, 0])

  for i in range(len(train[1:])):
      if i%50==0:
        print i
      im = cv2.imread('train-jpg/train_%s.jpg' % (int(i)))
      mean_green[i] = float(im[:, :, 1].mean())
      mean_red[i] = float(im[:, :, 0].mean())
      mean_blue[i] = float(im[:, :, 2].mean())
      std_green[i] = float(im[:, :, 1].std())
      std_red[i] = float(im[:, :, 0].std())
      std_blue[i] = float(im[:, :, 2].std())
  return np.r_['1,2,0', mean_green, mean_red, mean_blue, std_green, std_red, std_blue]
