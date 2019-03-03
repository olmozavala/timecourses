import numpy as np
import cv2
from os.path import join
import matplotlib.pyplot as plt

def getBorder(img):
    all_dims = img.shape
    mask = np.zeros(all_dims)
    th1 = 50
    th2 = 15
    for cur_col in range(all_dims[1]):
        # Searching bottom up
        for cur_row in range(all_dims[0]-1,200,-1):
            if img[cur_row,cur_col] > th2:
                mask[cur_row,cur_col] = 1
                break
        # Searching top bottom
        for cur_row in range(0,200):
            if img[cur_row,cur_col] > th1:
                mask[cur_row,cur_col] = 1
                break

    return mask


data_folder = '/media/osz1/DATA/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Data'
output_folder = '/media/osz1/DATA/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Output'
file_name = 'GD3T4control.avi'
cap = cv2.VideoCapture(join(data_folder,file_name))

mask = []
mult_pos = [[179,x] for x in np.arange(1500,1900,50)]
intensities = [[] for x in mult_pos]
size_box = [40,7]
frame_idx = 0
frame_rate = 24

while(cap.isOpened()):
    ret, frame = cap.read()
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(F'---Frame {frame_idx} failed: {e} ----')
        break

    mask = getBorder(img)

    # if (frame_idx % 20) == 0:
        # if frame_idx == 1:
    plt.imshow(img)
    plt.contour(mask, colors='r', linewidths=.3)
    file_name = join(output_folder,F'{frame_idx:02d}.png')
    plt.savefig(file_name)
    plt.show()

    frame_idx+=1

cap.release()
