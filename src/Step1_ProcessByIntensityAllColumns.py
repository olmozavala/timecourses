from os.path import join
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from FiltersAndUtils import *
from scipy.interpolate import *

def makeObj(name, interp_pts):

    vid = {
        'file_name': name, # Name of the file
        'interp_pts': interp_pts
    }

    return vid

def checkDirs(output_folder):
    cur_path = join(output_folder,'MaskArea')
    if not(os.path.exists(cur_path)):
        os.makedirs(cur_path)
    cur_path = join(output_folder,'Curves')
    if not(os.path.exists(cur_path)):
        os.makedirs(cur_path)
    cur_path = join(output_folder,'FilteredImages')
    if not(os.path.exists(cur_path)):
        os.makedirs(cur_path)
    cur_path = join(output_folder,'Original')
    if not(os.path.exists(cur_path)):
        os.makedirs(cur_path)


if __name__ == '__main__':
    # %%

    # THis is the first file we must run. It iterates over the frames of the videos, obtains the top and bottom
    # contours and saves the bottom position, the area and the mean intensities for each column

    # data_folder = '../Data/GD4'
    data_folder = '/media/osz1/DATA/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Data/GD4'
    output_folder = '../Output/GD4'
    checkDirs(output_folder)

    # %%
    videos=[]
    # ================= OLD ==================
    # videos.append(makeObj('ASalGD3M01H02ctrl2inj11AMdis3PM_3', 80, .83, .83, True))
    # videos.append(makeObj('GD2_11AM_2', 150, .8, 0, True))
    # videos.append(makeObj('GD3_11AM', 110, .8, 1.9, False))
    # videos.append(makeObj('GD3T4control', 140, .8, 0, True))
    # videos.append(makeObj('NP', 110, .6, 0, True))
    # videos.append(makeObj('RGD3T4M01H01', 90, .8, .7, False))
    # videos.append(makeObj('RDG3TEM01H01Sal1', 80, .83, .7, False))

    # ================= GD3 ==================
    # videos.append(makeObj('RDG3T4M01H01Sal1', -1, .83, .7, False))
    # videos.append(makeObj('RDG3T4M01H01Sal1', -1, .85, .6, False))
    # videos.append(makeObj('RDG3T4M01H02Sal1', -1, .8, .9, False))
    # videos.append(makeObj('RGD3T4M01H01', -1, .83, .7, False))
    # videos.append(makeObj('RGD3T4M01H02', -1, 1.10, .9, False))
    # videos.append(makeObj('RGD3T4M02H01', -1, .83, .7, False))
    # videos.append(makeObj('RGD3T4M02H01Sal', -1, .83, .7, False))
    # videos.append(makeObj('RGD3T4M02H02', -1, .8, .8, False))
    # videos.append(makeObj('RGD3T4M02H02Sal', -1, .8, .8, False))
    # videos.append(makeObj('RGD3T4M03H01', -1, 1.1, 1.1, False))
    # videos.append(makeObj('RGD3T4M03H01Sal', -1, .8, 1.0, False))
    # videos.append(makeObj('RGD3T4M03H02', -1, 1.1, 1, False))
    # videos.append(makeObj('RGD3T4M03H02Sal', -1, 1, 1, False))
    # videos.append(makeObj('RGD4T4M01H01', -1, 1, 1.3, False))
    # videos.append(makeObj('RGD4T4M01H01Sal2', -1, 1.3, 1.3, False))

    # ================= GD4 ==================
    # Order: name, mean_uterus_size, th_bot, th_top, only_bot):
    videos.append(makeObj('RGD4T4M01H01Sal2',20))
    videos.append(makeObj('RGD4T4M01H01',10)) # 8
    videos.append(makeObj('RGD4T4M01H02',20))
    videos.append(makeObj('RGD4T4M02H01MdSal2',10)) # 8
    videos.append(makeObj('RGD4T4M02H02MdSal2',20))
    videos.append(makeObj('RGD4T4M03H01Md',20))
    videos.append(makeObj('RGD4T4M03MdH01Sal2',10)) # 10
    videos.append(makeObj('RGD4T4M01H01Sal2',20))
    videos.append(makeObj('RGD4T4M01H02Sal2',20))
    videos.append(makeObj('RGD4T4M02H02Md',20))
    videos.append(makeObj('RGD4T4M02MdH01',10)) #7
    videos.append(makeObj('RGD4T4M03H02Md',20))
    videos.append(makeObj('RGD4T4M03MdH02Sal2',20))

    plot_every_n_frames = 20

    for i,cur_vid in enumerate(videos):
        file_name = cur_vid['file_name']
        print(F'Working with file {file_name}')
        try:
            cap = cv2.VideoCapture(join(data_folder,file_name+'.avi'))
            rows,cols,frames = getDims(cap) # Getting size of video
            cap.release()

            cap = cv2.VideoCapture(join(data_folder,file_name+'.avi'))
            all_video = np.zeros((frames, rows, cols))

            frame_idx = 0 # Index for each frame
            frame_rate = 10# Frame rate, to plot the proper time in each plot

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
            print('Done!')

        except Exception as e:
            print(F'---Failed for video {file_name} failed: {e} ----')

        print('Smoothing original frames!')
        g_size = 11 # Smoothing size
        smooth = np.zeros(all_video.shape)
        for cur_frame in range(frames):
            smooth[cur_frame,:,:] = cv2.GaussianBlur(all_video[cur_frame,:,:], (g_size,g_size), 0)
        for cur_col in range(cols):
            smooth[:,:,cur_col] = cv2.GaussianBlur(smooth[:,:,cur_col], (g_size,g_size), 0)

        print('Done!')

        top_pos = np.zeros((frames,cols))
        bottom_pos = np.zeros((frames,cols))
        mean_intensities = np.zeros((frames,cols))
        area_vals = np.zeros((frames,cols))

        print('Smoothing resulting positions to avoid spikes!')
        for cur_frame in range(frames):
            img = all_video[cur_frame,:,:]
            # Trying to use sobel. Amost there, but hte results are similar
            sobelx = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
            sobely = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

            c_sob = 0
            c_sob = 2*sobelx + sobely

            bottom_pos[cur_frame,:]= np.argmin(c_sob, axis=0)
            top_pos[cur_frame,:]= np.argmax(c_sob, axis=0)
            # if (cur_frame % plot_every_n_frames) == 0: # Only plot once every x frames
            #     plt.imshow(img)
            #     plt.title(F'{cur_frame}')
            #     plt.show()
            #     plt.imshow(c_sob)
            #     plt.title(F'{cur_frame}')
            #     plt.show()

            # In this case we need to fix the top values
            # if not(cur_vid['use_max_sob']):

            # ------------ Smoothing top positions -------------
            tot_pts = cur_vid['interp_pts']
            x = np.linspace(0,cols,tot_pts,dtype=np.int)
            x_range = int(x[2] - x[1])
            y = np.zeros(x.shape)
            c_x = int(x_range/2)
            y[0] = np.mean(top_pos[cur_frame,0:c_x])
            for x_idx in range(1,tot_pts-1):
                y[x_idx] = np.mean(top_pos[cur_frame,c_x:c_x+x_range])
                c_x+=x_range
            y[-1] = np.mean(top_pos[cur_frame,c_x:-1])

            f2 = interp1d(x, y, kind='cubic')
            xnew = np.linspace(0, cols, cols, dtype=np.int)

            # plt.plot(xnew, top_pos[cur_frame,:], '--')
            # plt.scatter(x, y)
            # plt.plot(xnew, f2(xnew), '--')
            # plt.show()
            top_pos[cur_frame,:] = f2(xnew)

            # ------------ Smoothing bottom positions -------------
            tot_pts = 35
            x = np.linspace(0,cols,tot_pts,dtype=np.int)
            x_range = int(x[2] - x[1])
            y = np.zeros(x.shape)
            c_x = int(x_range/2)
            y[0] = np.mean(bottom_pos[cur_frame,0:c_x])
            for x_idx in range(1,tot_pts-1):
                y[x_idx] = np.mean(bottom_pos[cur_frame, c_x:c_x+x_range])
                c_x += x_range
            y[-1] = np.mean(bottom_pos[cur_frame,c_x:-1])
            xnew = np.linspace(0, cols, cols, dtype=np.int)

            f2 = interp1d(x, y, kind='cubic')
            bottom_pos[cur_frame,:] = f2(xnew)

            # if cur_vid['from_half']:
            #     top_pos[cur_frame,int(cols/2):]= np.argmax(c_sob[:,int(cols/2):]>cur_vid['max_sob_value'], axis=0)
            # else:
            #     top_pos[cur_frame,:]= np.argmax(c_sob>cur_vid['max_sob_value'], axis=0)

        print('Done!')

        # Blurring the final positions
        top_pos = cv2.blur(top_pos, (10,10))
        bottom_pos = cv2.blur(bottom_pos, (10,10))

        bottom_pos = bottom_pos.astype(int)
        top_pos = top_pos.astype(int)

        print('Computing means and areas!')
        for cur_frame in range(frames):
            mask = np.zeros((rows,cols))
            for cur_col in range(cols):
                mask[top_pos[cur_frame,cur_col]:bottom_pos[cur_frame,cur_col],cur_col] =  1

            mean_intensities[cur_frame,:] = np.true_divide(all_video[cur_frame,:,:].sum(0), (mask!=False).sum(0))

            if (cur_frame % plot_every_n_frames) == 0: # Only plot once every x frames
                print(F"Frame {cur_frame}")
                plt.imshow(all_video[cur_frame,:,:])
                plt.title(F'{file_name} Frame:{cur_frame}')
                plt.contour(mask, colors='r', linewidths=.3)
                plt.savefig(join(output_folder,'MaskArea',F'{file_name}_Mask_frame_{cur_frame:04d}.jpg'),
                                bbox_inches='tight')
                plt.close()
                # plt.show()
        area_vals = bottom_pos - top_pos
        print('Done!')

        print('Saving results...')
        plt.matshow(mean_intensities)
        plt.title(F'{file_name}')
        plt.savefig(join(output_folder,F'{file_name}_Mean_intensities.jpg'), bbox_inches='tight')
        plt.close()

        plt.matshow(bottom_pos)
        plt.title(F'{file_name}')
        plt.savefig(join(output_folder,F'{file_name}_Bottom_positions.jpg'), bbox_inches='tight')
        plt.close()

        plt.matshow(top_pos)
        plt.title(F'{file_name}')
        plt.savefig(join(output_folder,F'{file_name}_Top_positions.jpg'), bbox_inches='tight')
        plt.close()

        plt.matshow(area_vals)
        plt.title(F'{file_name}')
        plt.savefig(join(output_folder,F'{file_name}_Area.jpg'), bbox_inches='tight')
        plt.close()

        np.savetxt(join(output_folder,F'{file_name}_Mean_intensities.csv'), mean_intensities,fmt='%10.3f', delimiter=',')
        np.savetxt(join(output_folder,F'{file_name}_Bottom_positions.csv'), bottom_pos,fmt='%10.3f', delimiter=',')
        np.savetxt(join(output_folder,F'{file_name}_Top_positions.csv'), top_pos,fmt='%10.3f', delimiter=',')
        np.savetxt(join(output_folder,F'{file_name}_Area.csv'), area_vals,fmt='%10.3f', delimiter=',')
        print('Done!!!')

