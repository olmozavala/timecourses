from os.path import join
import cv2
import matplotlib.pyplot as plt
import numpy as np
from FiltersAndUtils import *

def makeObj(name, mean_uterus_size, th_bot, th_top, only_bot):
    vid = {
        'file_name':name, # Name of the file
        'mean_ut_size':mean_uterus_size, # Mean apparent size of uterus in video
        'th_bot':th_bot,  # Threshold for searching bottom up wit respect to the mean intensity ( > 0)
        'th_top':th_top,  # Threshold for searching top bottom up wit respect to the mean intensity ( > 0)
        'only_bot':only_bot  # Indicates if we search only bottom up
    }
    return vid

if __name__ == '__main__':
    #%% Read data
    data_folder = '../Data'
    output_folder = '../Output'

    videos=[]
    videos.append(makeObj('ASalGD3M01H02ctrl2inj11AMdis3PM_3', 80, .83, .83, True))
    videos.append(makeObj('GD2_11AM_2', 150, .8, 0, True))
    videos.append(makeObj('GD3_11AM', 110, .8, 1.9, False))
    videos.append(makeObj('GD3T4control', 140, .8, 0, True))
    videos.append(makeObj('NP', 110, .6, 0, True))
    videos.append(makeObj('RGD3T4M01H01', 90, .8, .7, False))
    videos.append(makeObj('RDG3TEM01H01Sal1', 80, .83, .7, False))

    for i,cur_vid in enumerate(videos):
        file_name = cur_vid['file_name']
        print(F'Working with file {file_name}')
        try:
            cap = cv2.VideoCapture(join(data_folder,file_name+'.avi'))
            rows,cols,frames = getDims(cap) # Getting size of video
            cap.release()

            cap = cv2.VideoCapture(join(data_folder,file_name+'.avi'))

            plot_every_n_frames = 5
            mean_uterus_size = cur_vid['mean_ut_size']# Mean size of the uterus, used to select the size of the rectangle
            mean_intensities = np.zeros((frames, cols)) # Hard coded
            bot_positions = np.zeros((frames, cols)) # Hard coded
            top_positions = np.zeros((frames, cols)) # Hard coded
            area_positions = np.zeros((frames, cols)) # Hard coded

            frame_idx = 0 # Index for each frame
            frame_rate = 10# Frame rate, to plot the proper time in each plot

            #%% Iterate over each frame
            print('Obtaining mean intensities.....')
            while(cap.isOpened()):

                # Obtains a frame for each vide (specific CV structure)
                ret, frame = cap.read()
                try:
                    # Gets the frame as an RGB numpy matrix
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except Exception as e:
                    print(F'---Frame {frame_idx} failed: {e} ----')
                    break

                mask, mean_intensities[frame_idx,:],bot_positions[frame_idx,:],top_positions[frame_idx,:] = getROI(img, mean_uterus_size,
                                                                                cur_vid['th_bot'], cur_vid['th_top'], cur_vid['only_bot'])

                area_positions = bot_positions - top_positions
                if (frame_idx % plot_every_n_frames) == 0: # Only plot once every x frames
                    print(F"Frame {frame_idx}")
                    plt.imshow(img)
                    plt.contour(mask, colors='r', linewidths=.3)
                    plt.savefig(join(output_folder,'MaskArea',F'{file_name}_Mask_frame_{frame_idx:04d}.jpg'),
                                    bbox_inches='tight')
                    plt.close()

                # if frame_idx > 20:
                #     break
                frame_idx+=1

            print('Saving results...')
            plt.matshow(mean_intensities)
            plt.savefig(join(output_folder,F'{file_name}_Mean_intensities.jpg'), bbox_inches='tight')
            plt.close()

            plt.matshow(bot_positions)
            plt.savefig(join(output_folder,F'{file_name}_Bottom_positions.jpg'), bbox_inches='tight')
            plt.close()

            plt.matshow(top_positions)
            plt.savefig(join(output_folder,F'{file_name}_Top_positions.jpg'), bbox_inches='tight')
            plt.close()

            plt.matshow(area_positions)
            plt.savefig(join(output_folder,F'{file_name}_Area.jpg'), bbox_inches='tight')
            plt.close()

            np.savetxt(join(output_folder,F'{file_name}_Mean_intensities.csv'), mean_intensities,fmt='%10.3f', delimiter=',')
            np.savetxt(join(output_folder,F'{file_name}_Bottom_positions.csv'), bot_positions,fmt='%10.3f', delimiter=',')
            np.savetxt(join(output_folder,F'{file_name}_Top_positions.csv'), top_positions,fmt='%10.3f', delimiter=',')
            np.savetxt(join(output_folder,F'{file_name}_Area.csv'), area_positions,fmt='%10.3f', delimiter=',')
            print('Done!!!')
            cap.release()

        except Exception as e:
            print(F'---Failed for video {file_name} failed: {e} ----')
            break
