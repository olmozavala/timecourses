import cv2
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.signal import freqz

def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    Creates the coefficients of the band pass filter
    :param lowcut:
    :param highcut:
    :param fs: Sampling frequency
    :param order: Order of the filter
    :return:
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    ''' Filters the data'''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def smoothCurves(all_int, n):
    '''Smooths all the intensities by computing the average for every n points'''
    sm_int = all_int.copy()
    time_steps = len(all_int[0])
    for c_idx, c_point in enumerate(all_int):
        # Here we compute the mean value for all the 'intermediate' intensities for x-n to x+n
        sm_int[c_idx,n:time_steps-n] = [c_point[x-n:x+n].mean() for x in range(n,time_steps-n)]

    return sm_int

def smoothSingleCurve(all_int, n):
    '''Smooths all the intensities by computing the average for every n points'''
    sm_int = all_int.copy()
    time_steps = len(all_int)
    sm_int[n:time_steps-n] = [all_int[x-n:x+n].mean() for x in range(n,time_steps-n)]
    return sm_int

def getDims(cap):
    frame_idx = 0
    while(cap.isOpened()):
        # Obtains a frame for each vide (specific CV structure)
        ret, frame = cap.read()
        try:
            # if frame_idx == 0:
                # Gets the frame as an RGB numpy matrix
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows = img.shape[0]
            cols = img.shape[1]
            frame_idx += 1
        except Exception as e:
            print(F'---Frame {frame_idx} failed: {e} ----')
            break
    print(F'Frame dims: rows:{rows}, cols:{cols}, frames:{frame_idx}')
    return rows, cols, frame_idx

def getROI(img, mean_uterus_size, th_bot_all, th_top_all, only_bottom=True):
    '''What we do here is obtain the mean intensity values at all positions inside the uterus '''
    all_dims = img.shape
    border = 10 # Size of the border (we skip this part)
    mask = np.zeros(all_dims, dtype=bool)
    mask_int = np.zeros(all_dims)
    rows = all_dims[0]
    cols = all_dims[1]
    bot_pos = np.zeros(cols)
    top_pos = np.zeros(cols)
    mean_th = np.mean(img, axis=0) # Computes mean value from half the scrren to the top
    th_bot = mean_th*th_bot_all
    th_top = mean_th*th_top_all
    if only_bottom:
        for cur_col in range(cols):
            # Searching bottom up
            for cur_row in range(rows-1,0,-1):
                if img[cur_row,cur_col] > th_bot[cur_row]:
                    mask[cur_row-mean_uterus_size+border:cur_row-border,cur_col] = True
                    bot_pos[cur_col] = cur_row
                    break
    else:
       for cur_col in range(cols):
            # Searching bottom up
            for cur_row in range(rows-1,0,-1):
                if img[cur_row,cur_col] > th_bot[cur_row]:
                    end_pos = cur_row
                    bot_pos[cur_col] = cur_row
                    break

            # Searching top bottom
            for cur_row in range(0,rows):
                if img[cur_row,cur_col] > th_top[cur_row]:
                    start_pos = cur_row
                    top_pos[cur_col] = cur_row
                    break

            mask[start_pos+border:end_pos-border,cur_col] = True

    mask_int[mask] = img[mask]
    mean_values = np.true_divide(mask_int.sum(0), (mask!=False).sum(0))
    return mask, mean_values, bot_pos, top_pos
