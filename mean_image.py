import glob
import numpy as np
import cv2
import Tkinter,tkFileDialog
import sys,os

try:
    mode=sys.argv[1]
except:
    mode='mean'
root = Tkinter.Tk()
filez = tkFileDialog.askopenfilenames(parent=root,title='Choose a file')
fns=root.tk.splitlist(filez)
root.destroy()
wdir=fns[0].split('/')[:-1]
os.chdir('/'.join(wdir))
im1=cv2.imread(fns[0],-1)
rows,cols=int(im1.shape[0]),int(im1.shape[1])
if im1.shape[2]:
    all_im=np.zeros((rows,cols,3,len(fns))).astype(float)
    for n,i in enumerate(fns):
        add_im=cv2.imread(i,-1)
        all_im[:,:,:,n]=add_im
    if mode=='mean':
        tot_mean=np.mean(all_im, axis=3)
        if str(add_im.dtype)=='uint8':
            cv2.imwrite('mean_of_ims.tif', tot_mean.astype(np.uint8))
        else:
            cv2.imwrite('mean_of_ims.tif', tot_mean.astype(np.uint16))
    elif mode=='median':
        tot_mean=np.median(all_im, axis=3)
        if str(add_im.dtype)=='uint8':
            cv2.imwrite('mean_of_ims.tif', tot_mean.astype(np.uint8))
        else:
cv2.imwrite('mean_of_ims.tif', tot_mean.astype(np.uint16))
