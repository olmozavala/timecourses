import numpy as np
import cv2
from os.path import join
import matplotlib.pyplot as plt

def getBorderPos(img,cur_col):
    all_dims = img.shape
    th1 = 50
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

    return bottom_th, top_th

def getMeanIntensities(img, pos, size_box):
    ''' From a start position and box size, it obtains the mean intensity of an image'''
    all_values = img[pos[0]-size_box[0]:pos[0]+size_box[0],pos[1]-size_box[1]:pos[1]+size_box[1]]
    return all_values.mean()

def smoothCurves(all_int, n):
    '''Smooths all the intensities by computing the average for every n points'''
    sm_int = all_int.copy()
    time_steps = len(all_int[0])
    for c_idx, c_point in enumerate(all_int):
        # Here we compute the mean value for all the 'intermediate' intensities for x-n to x+n
        sm_int[c_idx,n:time_steps-n] = [c_point[x-n:x+n].mean() for x in range(n,time_steps-n)]

    return sm_int

# data_folder = '/media/osz1/DATA/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Data'
# output_folder = '/media/osz1/DATA/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Output'

# data_folder = '/home/olmozavala/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Data'
# output_folder = '/home/olmozavala/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Output'
data_folder = '../Data'
output_folder = '../Output'
file_name = 'GD3T4control.avi'
cap = cv2.VideoCapture(join(data_folder,file_name))

mask = [] # Will contain a mask for each rectangle
# This array contains the start positions of each rectangle (x,y)
y_pos = [[179,x] for x in np.arange(1800,1900,50)]
# This one will have the mean intensities for each rectangle in each frame
intensities = [[] for x in y_pos]
size_box = [40,10] # This is the size of each rectanngle (vertical, horizontal)
plot_every_n_frames = 100
mean_avg_size = 5 # How many times steps are we going to use for smoothing the curves

frame_idx = 0 # Index for each frame
frame_rate = 24 # Frame rate, to plot the proper time in each plot

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

    # For each position we create the mask and obtain the mean intensities
    for pos_idx, pos in enumerate(y_pos):
        mask[pos[0]-size_box[0]:pos[0]+size_box[0],pos[1]-size_box[1]:pos[1]+size_box[1]] = 1
        intensities[pos_idx].append(getMeanIntensities(img,pos,size_box))

    # Here we plot were the rectangles are
    # if (frame_idx % plot_every_n_frames) == 0: # Only plot once every x frames
    if False:
        plt.imshow(img)
        plt.contour(mask, colors='r', linewidths=.3)
        file_name = join(output_folder,F'Frame_{frame_idx:02d}.png')
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()

    frame_idx+=1

intensities = np.array(intensities) # Make the array a numpy array
print(F' The final size of our intensities array is {intensities.shape}')
print('Smoothing the curves...')
intensities = smoothCurves(intensities, int(np.floor(mean_avg_size/2)))

drawAllTogether = False
time = np.arange(len(intensities[0]))/frame_rate
for pos_idx, pos in enumerate(y_pos):
    title=F'Change of intensities at pos: {pos}'
    plt.title(title)
    plt.plot(time,intensities[pos_idx], label=F'Pos: {pos}')
    if not(drawAllTogether):
        file_name = join(output_folder,F'{pos[0]}_{pos[1]}.png')
        plt.xlabel('Seconds')
        plt.ylabel('Intensity')
        plt.grid()
        plt.legend(loc='best')
        plt.savefig(file_name)
        plt.show()

if drawAllTogether:
    file_name = join(output_folder,'All.png')
    plt.xlabel('Seconds')
    plt.ylabel('Intensity')
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(file_name)
    plt.show()

cap.release()
