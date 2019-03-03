import numpy as np
import cv2
from os.path import join
import matplotlib.pyplot as plt

def getMeanIntensities(img, pos, size_box):
    all_values = img[pos[0]-size_box[0]:pos[0]+size_box[0],pos[1]-size_box[1]:pos[1]+size_box[1]]
    return all_values.mean()


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

    if frame_idx == 0:
        mask = np.zeros((img.shape))

    for pos_idx, pos in enumerate(mult_pos):
        mask[pos[0]-size_box[0]:pos[0]+size_box[0],pos[1]-size_box[1]:pos[1]+size_box[1]] = 1
        intensities[pos_idx].append(getMeanIntensities(img,pos,size_box))

    if (frame_idx % 20) == 0:
    # if frame_idx == 1:
        plt.imshow(img)
        plt.contour(mask, colors='r', linewidths=.3)
        file_name = join(output_folder,F'{frame_idx:02d}.png')
        plt.savefig(file_name)
        plt.show()

    frame_idx+=1


time = np.arange(len(intensities[0]))/frame_rate
for pos_idx, pos in enumerate(mult_pos):
    title=F'Change of intensities at pos: {pos}'
    plt.title(title)
    plt.plot(time,intensities[pos_idx])
    plt.xlabel('Seconds')
    plt.ylabel('Intensity')
    plt.grid()
    file_name = join(output_folder,F'{pos[0]}_{pos[1]}.png')
    plt.savefig(file_name)
    plt.show()

cap.release()
