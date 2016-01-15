import numpy as np
import cv2
import glob
from skimage import io
import matplotlib.pyplot as plt
import PIL.Image
from cStringIO import StringIO
import IPython.display


iml=io.imread('http://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png')
imr=io.imread('http://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png')

###Can change low variable to change thresholding result

low,high=177,255

imgrayl = iml[:,:,1]
imgrayr = imr[:,:,1]#Make grayscale

_,threshl = cv2.threshold(imgrayl,low,high,0) #Detect edges
_,threshr = cv2.threshold(imgrayr,low,high,0)

####Variables we can change for contour search - maxarea, minarea, contournumber

#Contour number = size of contour in search space, 1=Biggest contour, 2=Second Biggest etcetc

minarea,maxarea,contour=0,100000000,1

ims=[iml,imr] # original images
ims_in=[threshl,threshr]

for num,thresh in enumerate(ims_in):
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #Find contour method. 

    #In openCV 3.0 - _,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    areas=[]
    for n,i in enumerate(contours): #loop to find area
        try:
            area=cv2.contourArea(i)
        except:
            '' #Exceptions for no contours
        areas.append(area)

    areas=np.array(areas)
    cross=np.intersect1d(np.where(areas<maxarea)[0],np.where(areas>minarea)[0]) #Find positions where query conditions are met
    cont=np.where(areas[cross]==np.sort(areas[cross])[-contour])[0] #Sort by size, find the i-th biggest in the query
    cnts2=np.array(contours)[cross]
    rect=cv2.minAreaRect(cnts2[cont][0]) #find the rectangle of minimum area of the contour

    box = cv2.cv.BoxPoints(rect) #Find those points
    box = [[np.array(i).astype(int).tolist()] for i in box]
    box=np.array(box)

    mask=np.zeros_like(imgrayl)
    mask_box=np.zeros_like(imgrayl)

    cv2.drawContours(mask, np.array(contours)[cross], cont, 1, -1) #draw it
    cv2.drawContours(mask_box, [box], 0, 1, -1) #draw it

    mask=np.tile(np.expand_dims(mask, axis=2), (1,1,3))
    mask_box=np.tile(np.expand_dims(mask_box, axis=2), (1,1,3))
