import numpy as np
import cv2
from os.path import join
import matplotlib.pyplot as plt
from FiltersAndUtils import *

def getBorderPos(img,cur_col):
    all_dims = img.shape
    th1 = 27
    th2 = 15
    # Searching bottom up
    for cur_row in range(all_dims[0]-1,200,-1):
        if img[cur_row,cur_col] > th2:
            bottom_th = cur_row
            break
    # Searching top bottom
    for cur_row in range(0,200):
        if img[cur_row,cur_col] > th1:
            top_th= cur_row
            break

    return top_th, bottom_th

def getMeanIntensities(img, pos, bbox):
    ''' From a start position and box size, it obtains the mean intensity of an image'''
    all_values = img[pos[0]-bbox[0]:pos[0]+bbox[0],pos[1]-bbox[1]:pos[1]+bbox[1]]
    return all_values.mean()

# data_folder = '/media/osz1/DATA/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Data'
# output_folder = '/media/osz1/DATA/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Output'

# data_folder = '/home/olmozavala/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Data'
# output_folder = '/home/olmozavala/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Output'
data_folder = '../Data'
output_folder = '../Output'
file_name = 'GD3_11AM.avi'
cap = cv2.VideoCapture(join(data_folder,file_name))

mask = [] # Will contain a mask for each rectangle
# This array contains the start positions of each rectangle (x,y)
x_pos = np.arange(600,1100,40)
mean_uterus_size = 120
# This one will have the mean intensities for each rectangle in each frame
intensities = [[] for x in x_pos]
bbox = {'x':10,'y':mean_uterus_size-50} # This is the size of each rectanngle (vertical, horizontal)
plot_every_n_frames = 3
mean_avg_size = 3 # How many times steps are we going to use for smoothing the curves

frame_idx = 0 # Index for each frame
frame_rate = 24 # Frame rate, to plot the proper time in each plot

print('Obtaining mean intensities.....')
# Iterates over the video
while(cap.isOpened()):

    # Obtains a frame for each vide (specific CV structure)
    ret, frame = cap.read()
    try:
        # Gets the frame as an RGB numpy matrix
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(F'---Frame {frame_idx} failed: {e} ----')
        break

    # If we are in the first frame we initialize the mask array
    if frame_idx == 0:
        mask = np.zeros((img.shape))
    mask[:,:] = 0

    # For each position we create the mask and obtain the mean intensities
    for pos_idx, x_start in enumerate(x_pos):
        y_top, y_bottom = getBorderPos(img,x_start)
        # Only using the y_bottom because is more robust
        y_middle = y_bottom - mean_uterus_size/2
        x_end = int(x_start + bbox['x'])
        y_start = int(y_middle - bbox['y']/2)
        y_end = int(y_middle + bbox['y']/2)
        # mask[y_bottom, x_start] = 2
        # mask[y_top, x_start] = 2
        mask[y_start:y_end, x_start:x_end] = 1
        values = img[mask > 0]
        intensities[pos_idx].append(values.mean())

    # Here we plot were the rectangles are
    # if (frame_idx % plot_every_n_frames) == 0: # Only plot once every x frames
    if frame_idx == 0:
        plt.imshow(img)
        plt.contour(mask, colors='r', linewidths=.3)
        file_name = join(output_folder,F'Frame_{frame_idx:02d}.png')
        # plt.savefig(file_name, bbox_inches='tight')
        plt.show()

    frame_idx+=1

print(F'Total number of frames {frame_idx}')
intensities = np.array(intensities) # Make the array a numpy array
print(F' The final size of our intensities array is {intensities.shape}')
print('Smoothing the curves...')
low_freq = smoothCurves(intensities, 20) # Gets low frequencies
removed_low = intensities - low_freq
clean_intensities = smoothCurves(removed_low, 2) # Removes high frequencies

drawAllTogether = True
time = np.arange(len(intensities[0]))/frame_rate
for pos_idx, cur_x_pos in enumerate(x_pos):
    title=F'Change of intensities at pos: {cur_x_pos}'
    plt.title(title)
    plt.plot(time,clean_intensities[pos_idx], label=F'Pos: {cur_x_pos}')
    if not(drawAllTogether):
        file_name = join(output_folder,F'{cur_x_pos}.png')
        plt.xlabel('Seconds')
        plt.ylabel('Intensity')
        plt.grid()
        plt.ylim([-5,5])
        plt.savefig(file_name)
        plt.show()

if drawAllTogether:
    file_name = join(output_folder,'All.png')
    plt.xlabel('Seconds')
    plt.ylabel('Intensity')
    plt.grid()
    # plt.legend(loc='best')
    plt.savefig(file_name)
    plt.show()

cap.release()
