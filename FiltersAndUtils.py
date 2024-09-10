import cv2
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.interpolate import *

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

def cubicSplines(data, final_pts):
    orig_pts = data.shape[0]
    x = np.linspace(0,orig_pts-1,final_pts)
    dx = x[1] - x[0]  # Real float dx
    hdx = dx/2 # Half dx
    y = np.zeros(x.shape)
    y[0] = np.mean(data[0:int(hdx)]) # Just for the first and last points we obtain the average for half the range.
    # print(F'0-{int(dx)} --> {y[0]}')
    c_x = hdx  # Current x position
    for x_idx in range(1,final_pts-1):
        y[x_idx] = np.mean(data[max(int(c_x-hdx),0):int(c_x+dx)])
        # print(F'{max(int(c_x-hdx),0)}-{int(c_x+hdx)} --> {y[x_idx]}')
        c_x += dx

    y[-1] = np.mean(data[int(c_x):-1])
    # print(F'{int(c_x)}-end --> {y[-1]}')

    f2 = interp1d(x, y, kind='cubic')
    xnew = np.arange(orig_pts)

    # plt.scatter(xnew, data,c='g')
    # plt.plot(x,y,'b')
    # plt.plot(xnew,f2(xnew),'r--')
    # plt.show()

    return f2(xnew)


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

def gaussianBlurXandZ(all_video, k_size_time, k_size_hor):
    '''Performs gaussian blur two times, one by each Frame and one by each column '''
    print(f"Smoothing video with a kernel size of time:{k_size_time} and horizontal:{k_size_hor}")
    frames,rows,cols = all_video.shape
    smooth = np.zeros(all_video.shape)
    for cur_col in range(cols):
        smooth[:,:,cur_col] = cv2.GaussianBlur(all_video[:,:,cur_col], (k_size_time, k_size_time), 0)
    for cur_frame in range(frames):
        smooth[cur_frame,:,:] = cv2.GaussianBlur(smooth[cur_frame,:,:], (k_size_hor, k_size_hor), 0)

    return smooth

def computeEdgesSobel(img,kernel_size):
    # Trying to use sobel. Amost there, but hte results are similar
    sobelx = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernel_size)
    sobely = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernel_size)

    c_sob = 0
    c_sob = 2*sobelx + sobely

    return c_sob

def readFramesFromVideoFile(file_name, ):
    '''Reads an avi file into a numpy array '''
    print(F'Working with file {file_name}')
    try:
        cap = cv2.VideoCapture(file_name)
        rows, cols, frames = getDims(cap) # Getting size of video
        cap.release()

        cap = cv2.VideoCapture(file_name)
        all_video = np.zeros((frames, rows, cols))

        # Printing info 
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / frame_rate

        print("Frame Count:", frame_count)
        print("Frame Width:", frame_width)
        print("Frame Height:", frame_height)
        print("Frame Rate:", frame_rate)
        print("Duration (seconds):", duration)


        frame_idx = 0 # Index for each frame

        print('Reading data...')
        while(frame_idx < frames):
            # Obtains a frame for each vide (specific CV structure)
            ret, frame = cap.read()
            try:
                # Gets the frame as an gray scale numpy matrix
                all_video[frame_idx,:,:] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_idx+=1

            except Exception as e:
                print(F'---Frame {frame_idx} failed: {e} ----')
                frame_idx+=1
                continue

        cap.release()

    except Exception as e:
        print(F'---Failed for video {file_name} failed: {e} ----')

    return all_video, rows, cols, frames

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

def computeExtentSpaceTime(img):
    """
    Computes the extent of the image in milimeters
    """
    fps = 5
    pix_per_mil = 56

    frames, pixels = img.shape
    miny = 0
    maxy = frames/fps

    minx = 0
    maxx = pixels/pix_per_mil

    return [minx,maxx,maxy,miny]


def computeExtentSpace(img):
    """
    Computes the extent of the image in milimeters
    """
    pix_per_mil_x = 110.85
    pix_per_mil_y = 110.85

    pixels_x, pixels_y = img.shape

    minx = 0
    maxx = pixels_y/pix_per_mil_y
    miny = 0
    maxy = pixels_x/pix_per_mil_x

    return [minx,maxx,maxy,miny]
