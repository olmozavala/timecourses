from os.path import join
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt, argrelextrema
from FiltersAndUtils import *
from Utils_Visualization import *
from Utils_io import *

#%%

# cubi_spline_pts_top = cols/10
def makeObj(name, top_edge_th, bottom_edge_th,
            median_filter_size_top=5,
            median_filter_size_bottom=5,
            cubic_splines_pts_top=100,
            cubic_splines_pts_bottom=100):

    vid = {
        'file_name': name, # Name of the file
        'top_edge_th': top_edge_th,
        'bottom_edge_th': bottom_edge_th,
        'medial_filter_size_top': median_filter_size_top,
        'medial_filter_size_bottom': median_filter_size_bottom,
        'cubic_splines_pts_top': cubic_splines_pts_top,
        'cubic_splines_pts_bottom': cubic_splines_pts_bottom
    }

    return vid

def checkDirs(output_folder):
    saveDir(join(output_folder,'MaskArea'))
    saveDir(join(output_folder,'Curves'))
    saveDir(join(output_folder,'FilteredImages'))
    saveDir(join(output_folder,'Original'))


if __name__ == '__main__':
    # %%

    # THis is the first file we must run. It iterates over the frames of the videos, obtains the top and bottom
    # contours and saves the bottom position, the area and the mean intensities for each column

    videos_path = 'GD3'
    data_folder = F'/home/olmozavala/Dropbox/MyProjects/DianaProjects/ContractionsFromVideos/Data/{videos_path}'
    output_folder = F'/data/DianaVideosOutput/{videos_path}'
    checkDirs(output_folder)
    disp_images = False # Indicates if we want to see the images as they are processed

    videos=[]

    # def_top_th = 1600
    # def_bot_th = 1900
    def_top_th = 1.8 
    def_bot_th = 1.8 
    plot_every_n_frames = 10
    k_size_time = 3# Used to smooth the vidoes (kernel size of a gaussian filter in time)
    k_size_space = 5# Used to smooth the vidoes (kernel size of a gaussian filter in time)

    if videos_path == 'GD3':
        # ================= GD3 ==================
        videos.append(makeObj('RGD3T4M01H01', def_top_th, def_bot_th))
        # videos.append(makeObj('RDG3T4M01H01Sal1',def_top_th,def_bot_th))
        # videos.append(makeObj('RGD3T4M01H02',def_top_th*1.1,def_bot_th*.8))
        # videos.append(makeObj('RDG3T4M01H02Sal1',def_top_th*1.5,def_bot_th*.8))
        # videos.append(makeObj('RGD3T4M02H01',def_top_th,def_bot_th))
        # videos.append(makeObj('RGD3T4M02H01Sal',def_top_th*.9,def_bot_th))
        # videos.append(makeObj('RGD3T4M02H02',def_top_th,def_bot_th*1.5))
        # videos.append(makeObj('RGD3T4M02H02Sal',def_top_th,def_bot_th))
        # videos.append(makeObj('RGD3T4M03H01',def_top_th*.7,def_bot_th))
        # videos.append(makeObj('RGD3T4M03H01Sal',def_top_th*.7,def_bot_th*.8))
        # videos.append(makeObj('RGD3T4M03H02',def_top_th,def_bot_th))
        # videos.append(makeObj('RGD3T4M03H02Sal',def_top_th*.9,def_bot_th*.8))
        # videos.append(makeObj('RGD4T4M01H01',def_top_th*.6,def_bot_th*.7))
        # videos.append(makeObj('RGD4T4M01H01Sal2',def_top_th*.55,def_bot_th*.45))
        # videos.append(makeObj('RGD3T4M06H01_2',def_top_th*1.5,def_bot_th*.45))
        # videos.append(makeObj('RGD3T4M06H01Sal_2',def_top_th,def_bot_th))
        # videos.append(makeObj('RGD3T4M06H02_2',def_top_th*.5,def_bot_th))
        # videos.append(makeObj('RGD3T4M06H02Sal_2',def_top_th*1.5,def_bot_th*1.2))
        # videos.append(makeObj('RGD3T4M07H01_2',def_top_th,def_bot_th))
        # videos.append(makeObj('RGD3T4M07H01Sal_2',def_top_th,def_bot_th))
        # videos.append(makeObj('RGD3T4M07H02_2',def_top_th,def_bot_th))
        # videos.append(makeObj('RGD3T4M07H02Sal_2',def_top_th,def_bot_th))
    else:
        # ================= GD4 ==================
        # Order: name, mean_uterus_size, th_bot, th_top, only_bot):
        videos.append(makeObj('RGD4T4M01H01Sal2',def_top_th,def_bot_th*.6))
        videos.append(makeObj('RGD4T4M01H01',def_top_th*.7,def_bot_th*.6))
        videos.append(makeObj('RGD4T4M01H02',def_top_th*.7,def_bot_th*.7))
        videos.append(makeObj('RGD4T4M02H01MdSal2',def_top_th*.8,def_bot_th*.8))
        videos.append(makeObj('RGD4T4M02H02MdSal2',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M03H01Md',def_top_th*.35,def_bot_th*.55))
        videos.append(makeObj('RGD4T4M03MdH01Sal2',def_top_th*.45,def_bot_th*.5)) # 10
        videos.append(makeObj('RGD4T4M01H02Sal2',def_top_th,def_bot_th*.85))
        videos.append(makeObj('RGD4T4M02H02Md',def_top_th*.85,def_bot_th))
        videos.append(makeObj('RGD4T4M02MdH01',def_top_th*.8,def_bot_th*.9)) #7
        videos.append(makeObj('RGD4T4M03H02Md',def_top_th*.45,def_bot_th*.7))
        videos.append(makeObj('RGD4T4M03MdH02Sal2',def_top_th*.5,def_bot_th*.6))
        videos.append(makeObj('RGD4T4M06H01_2',def_top_th*1.2,def_bot_th*.6))
        videos.append(makeObj('RGD4T4M06H01Sal_2',def_top_th,def_bot_th*.6))
        videos.append(makeObj('RGD4T4M06H02_2',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M06SalH02_2',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M07H01_2',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M07H02_2',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M07SalH01_2',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M07SalH02_2',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M08H01_2',def_top_th,def_bot_th*.8))
        videos.append(makeObj('RGD4T4M08H02_2',def_top_th*1.2,def_bot_th*1.1))
        videos.append(makeObj('RGD4T4M08H02Sal_2',def_top_th*1.5,def_bot_th))
        videos.append(makeObj('RGD4T4M08SalH01_2',def_top_th*1.1,def_bot_th*1.2))
        videos.append(makeObj('RGD4T4M09H01_2',def_top_th,def_bot_th*.8))
        videos.append(makeObj('RGD4T4M09H01_3',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M09H02_2',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M09H02_3',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M09SalH01_2',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M09SalH01_3',def_top_th,def_bot_th))
        videos.append(makeObj('RGD4T4M09SalH02_2',def_top_th,def_bot_th*.8))
        videos.append(makeObj('RGD4T4M09SalH02_3',def_top_th,def_bot_th))


    for i, cur_vid in enumerate(videos):
        try:
            file_name = cur_vid['file_name']
            print(F'******** {file_name} *************')
            pts_per_spline_top = cur_vid['cubic_splines_pts_top']
            pts_per_spline_bottom = cur_vid['cubic_splines_pts_bottom']

            top_edge_th = cur_vid['top_edge_th']
            bottom_edge_th = cur_vid['bottom_edge_th']

            median_filt_size_top = cur_vid['medial_filter_size_top']
            median_filt_size_bottom = cur_vid['medial_filter_size_bottom']

            print('\tReading Data....')
            all_video, rows, cols, frames = readFramesFromVideoFile(join(data_folder,file_name+'.avi'))
            # frames = 100  # Just to make it faster for debugging purposes
            # all_video = all_video[0:frames,:,:]
            cubi_spline_pts_top = int((cols/1000) * pts_per_spline_top)
            cubi_spline_pts_bottom = int((cols/1000) * pts_per_spline_bottom)
            print(F'Number of cubic splines points: {cubi_spline_pts_top}')
            print('\tDone!')

            print('\tSmoothing Data...')
            # Blurs the image in the X and Time dimensions
            # plotMultipleImages([all_video[0,:,:]], ['Temp'])
            smooth = gaussianBlurXandZ(all_video, k_size_time, k_size_space)
            plotMultipleImages([all_video[0,:,:], smooth[0,:,:]], ['Original', 'Smoothed'], view_results=disp_images)
            print('\tDone!')

            top_pos = np.zeros((frames,cols))
            bottom_pos = np.zeros((frames,cols))
            mean_intensities = np.zeros((frames,cols))
            area_vals = np.zeros((frames,cols))

            print('\tComputing top and bottom positions....')
            for cur_frame in range(frames):
                # Print the current frame every 100 frames
                if cur_frame % 100 == 0:
                    print(F'\t\tFrame: {cur_frame}')

                img = smooth[cur_frame,:,:]
                c_sob = computeEdgesSobel(img, 5)
                # Normalize sobel
                c_sob = c_sob / np.max(c_sob)

                # plotMultipleImages([smooth[0,:,:], c_sob], ['Smoothed Frame', 'Edge index from Sobel'], view_results=False,
                #                    extent=computeExtentSpace(smooth[0,:,:]), units=['Size in mm', 'Size in mm'],
                #                    output_folder=output_folder,
                #                     file_name = file_name,
                #                    cbar_label=['Intensity', 'Edge index from Sobel'])

                # Selects the top position when the 'cumulative' edges overpass a threshold (initial guess)
                top_pos[cur_frame,:] = np.argmax(np.cumsum(c_sob, axis=0) > top_edge_th,axis=0)
                bottom_pos[cur_frame,:] = rows - np.argmax(np.cumsum(np.flip(c_sob,axis=0), axis=0) 
                                                           < -bottom_edge_th, axis=0)

                # for c_col in range(50,100,5):
                #     fi = plt.figure(figsize=(10,5))
                #     plt.scatter(range(len(c_sob[:,c_col])), c_sob[:,c_col], s=1)
                #     plt.title(f"Column: {c_col}")
                #     plt.savefig(join(output_folder,f"5_{file_name}_col_{c_col}"), bbox_inches='tight')
                #     plt.close()

                # top_pos[cur_frame,:] = np.argmax(c_sob, axis=0)
                # bottom_pos[cur_frame,:] = rows - np.argmax(np.cumsum(np.flip(c_sob,axis=0), axis=0) < -bottom_edge_th, axis=0)

                # Computes all the local maxima in the image, only for half of the frame rows. 
                # comp_func = np.greater
                # indices = argrelextrema(c_sob, comp_func, axis=0, order=2)

                # # For each column we identify the local maxima and then we select the one with the highest value
                # print("Computing the locations...")
                # cumsum = np.cumsum(c_sob, axis=0)
                # for c_col in np.unique(indices[1]):
                #     row_indices = indices[0][indices[1] == c_col][:10] # Only consider the first 3 indexes
                #     max_value = np.max(c_sob[:, c_col])
                #     row_indices = np.append(row_indices, np.where(c_sob[:, c_col] == max_value)[0][0])

                #     maxima_values = c_sob[row_indices, c_col]
                #     sort_values_idx = np.argsort(maxima_values)
                #     # Compare the top two values, if the difference between the two is greater than %30 of the maxima value
                #     # then choose the highest, otherwise choose the one with the smallest index

                #     # Only plot everyt 10 frames
                #     if c_col % 10 == 0:
                #         fi = plt.figure(figsize=(10,5))
                #         plt.scatter(range(len(c_sob[:,c_col])), c_sob[:,c_col], s=1)
                #         plt.scatter(row_indices, maxima_values, c='r', s=2)
                #         plt.title(f"Column: {c_col}")
                #         plt.savefig(join(output_folder,f"5_{file_name}_col_{c_col}"), bbox_inches='tight')
                #         plt.close()

                #     if len(sort_values_idx) > 0:
                #         if (maxima_values[sort_values_idx[-1]] - maxima_values[sort_values_idx[-2]]) > (maxima_values[sort_values_idx[-1]] * 0.2):
                #             top_pos[cur_frame, c_col] = row_indices[sort_values_idx[-1]]
                #         else:
                #             top_pos[cur_frame, c_col] = np.min([row_indices[sort_values_idx[-2]], row_indices[sort_values_idx[-1]]])
                #     else:
                #         print(f"ERROR No maxima found for column: {c_col}")


                # Plot the cumulative sum of the column in the middle of c_sob
                if False:
                    plotMultipleImages([c_sob], ['Edge index from Sobel'], view_results=False,
                                   extent=computeExtentSpace(smooth[0,:,:]), units=['Size in mm', 'Size in mm'],
                                   output_folder=output_folder,
                                    file_name = f"4_{file_name}",
                                   cbar_label=['Edge index from Sobel'])

                    # pos = int(cols/2)
                    pos = 360
                    plt.plot(np.cumsum(c_sob, axis=0)[:, pos], zorder=1)
                    # plt.plot(c_sob[:, pos])
                    plt.scatter(top_pos[cur_frame,pos]-1, def_top_th, c='r', s=10, zorder=2, label='Selected top position')
                    plt.xlabel('Image Row')
                    plt.ylabel('Cummulative sum')
                    plt.title("Cumulative sum of the edge index")
                    plt.legend()
                    plt.savefig(join(output_folder,f"2_{file_name}"), bbox_inches='tight')
                    plt.close()
                    # plt.show()
                    plt.plot(np.cumsum(np.flip(c_sob,axis=0), axis=0)[:, pos])
                    plt.scatter(c_sob.shape[0] - bottom_pos[cur_frame,pos]-1, -def_bot_th, c='r', s=10, zorder=2, label='Selected bottom position')
                    plt.ylabel('Cummulative sum')
                    plt.xlabel('Image Row')
                    plt.title("Flipped cumulative sum of the edge index")
                    plt.legend()
                    # plt.show()
                    plt.savefig(join(output_folder,f"3_{file_name}"), bbox_inches='tight')
                    plt.close()

                # Plot before computing the median filter
                # if (cur_frame % plot_every_n_frames) == 0:
                #     plotImageAndScatter(all_video[cur_frame, :, :], [top_pos[cur_frame, :], bottom_pos[cur_frame, :]], 
                #                         title=F'Initially detected uterine horn',
                #                         units=['Frame column', 'Frame row'],
                #                         savefig=True, output_folder=join(output_folder,'MaskArea',file_name),
                #                         file_name = F'{file_name}_frame_{cur_frame:04d}_BeforeMedian.jpg', view_results=False)

                # ------------ Median filter on the obtained curves --------
                top_pos[cur_frame,:] = medfilt(top_pos[cur_frame], median_filt_size_top)
                bottom_pos[cur_frame,:] = medfilt(bottom_pos[cur_frame], median_filt_size_bottom)

                # For plotting BEFORE the smoothing
                if (cur_frame % plot_every_n_frames) == 0:
                    plotImageAndScatter(all_video[cur_frame, :, :], [top_pos[cur_frame, :], bottom_pos[cur_frame, :]], 
                                        title=F'Median filtered detected uterine horn',
                                        savefig=True, output_folder=join(output_folder,'MaskArea',file_name),
                                        file_name = F'{file_name}_frame_{cur_frame:04d}_BeforeSCubic.jpg', view_results=disp_images)

                # ------------ Smoothing curve of top positions -------------
                top_pos[cur_frame,:] = cubicSplines(top_pos[cur_frame,:], cubi_spline_pts_top)
                # ------------ Smoothing curve of bottom positions -------------
                bottom_pos[cur_frame,:] = cubicSplines(bottom_pos[cur_frame,:], cubi_spline_pts_bottom)

                # Plots the obtained mask and edges, only once every plot_every_n_frames frames
                if (cur_frame % plot_every_n_frames) == 0: # Only plot once every x frames
                    # plotImageAndScatter(all_video[cur_frame, :, :], [top_pos[cur_frame, :], bottom_pos[cur_frame, :]], title=F'{file_name} {cur_frame}',
                    plotImageAndScatter(all_video[cur_frame, :, :], [top_pos[cur_frame, :], bottom_pos[cur_frame, :]], 
                                        title=F'Uterine horn final detection. Time {cur_frame/5:0.1f} s',
                                        extent=computeExtentSpace(all_video[cur_frame, :, :]), 
                                        units=['Size in mm', 'Size in mm'],
                              savefig=True, output_folder=join(output_folder,'MaskArea',file_name),
                              file_name = F'{file_name}_frame_{cur_frame:04d}.jpg', view_results=disp_images)
                    if cur_frame == 0:
                        plotMultipleImages([c_sob], output_folder=join(output_folder, 'MaskArea', file_name),
                                           file_name='EdgesExample.jpg', view_results=disp_images)


            print('Done!')

            # Blurring the final positions
            # top_pos = cv2.blur(top_pos, (10,10))
            # bottom_pos = cv2.blur(bottom_pos, (10,10))

            bottom_pos = bottom_pos.astype(int)
            top_pos = top_pos.astype(int)

            print('Computing means intensities and areas!')
            for cur_frame in range(frames):
                mask = np.zeros((rows,cols))
                for cur_col in range(cols):
                    mask[top_pos[cur_frame,cur_col]:bottom_pos[cur_frame,cur_col],cur_col] =  1

                mean_intensities[cur_frame,:] = np.true_divide(all_video[cur_frame,:,:].sum(0), (mask != False).sum(0))
                # if (cur_frame % plot_every_n_frames) == 0: # Only plot once every x frames
                #     plotImageAndMask(all_video[cur_frame,:,:],mean_intensities[cur_frame,:],savefig=True,
                #                         output_folder=join(output_folder,'Mask_Area'),
                #                         file_name = F'{file_name}_Mask_frame_{cur_frame:04d}.jpg')

            area_vals = bottom_pos - top_pos
            print('Done!')

            print('Saving results...')
            final_folder = join(output_folder,'Original',file_name)
            saveDir(final_folder)

            plt.matshow(mean_intensities)
            plt.title(F'{file_name} Mean Intensities')
            plt.savefig(join(final_folder,F'{file_name}_Mean_intensities.jpg'), bbox_inches='tight')
            plt.close()

            plt.matshow(bottom_pos)
            plt.title(F'{file_name} Bottom Positions')
            plt.savefig(join(final_folder,F'{file_name}_Bottom_positions.jpg'), bbox_inches='tight')
            plt.close()

            plt.matshow(top_pos)
            plt.title(F'{file_name} Top Positions')
            plt.savefig(join(final_folder,F'{file_name}_Top_positions.jpg'), bbox_inches='tight')
            plt.close()

            plt.matshow(area_vals)
            plt.title(F'{file_name} Area')
            plt.savefig(join(final_folder,F'{file_name}_Area.jpg'), bbox_inches='tight')
            plt.close()

            np.savetxt(join(final_folder,F'{file_name}_Mean_intensities.csv'), mean_intensities,fmt='%10.3f', delimiter=',')
            np.savetxt(join(final_folder,F'{file_name}_Bottom_positions.csv'), bottom_pos,fmt='%10.3f', delimiter=',')
            np.savetxt(join(final_folder,F'{file_name}_Top_positions.csv'), top_pos,fmt='%10.3f', delimiter=',')
            np.savetxt(join(final_folder,F'{file_name}_Area.csv'), area_vals,fmt='%10.3f', delimiter=',')
            print('Done!!!')
        except Exception as e:
            print(F' ERROR failed for {cur_vid}: {e}')
