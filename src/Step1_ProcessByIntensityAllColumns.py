# %%
from os.path import join
import shutil
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    saveDir(join(output_folder, 'MaskArea'))
    saveDir(join(output_folder, 'Curves'))
    saveDir(join(output_folder, 'FilteredImages'))
    saveDir(join(output_folder, 'Original'))


# %% ----------------------------------- MAIN ---------------------------------

# THis is the first file we must run. It iterates over the frames of the videos, obtains the top and bottom
# contours and saves the bottom position, the area and the mean intensities for each column

# videos_path = 'GD3'
videos_path = 'SpecificWaves'
data_folder = F'/data/RiplaUterusWaves/Videos/{videos_path}'
output_folder = F'/data/RiplaUterusWaves/Outputs/{videos_path}OneThirdBottomIntensities'
disp_images = False # Indicates if we want to see the images as they are processed
frame_to_plot = 235# Which frame to generate plots for the paper 

# def_top_th = 1600
# def_bot_th = 1900
def_top_th = 2
def_bot_th = 2
plot_every_n_frames = 40
k_size_time = 3# Used to smooth the vidoes (kernel size of a gaussian filter in time)
k_size_space = 5# Used to smooth the vidoes (kernel size of a gaussian filter in time)

# Store all videos in data_folder ending with .avi
# all_files = os.listdir(data_folder)
# sort the files
# all_files.sort()

videos = []
videos.append(makeObj("BPM1-H1-PGE2D1-2.avi", def_top_th, def_bot_th))
videos.append(makeObj("BPM2-H2-PGE2D1-2.avi", def_top_th, def_bot_th))
videos.append(makeObj("BPM3-H1-CONTROL-1.avi", def_top_th, def_bot_th))
videos.append(makeObj("CM3-BL6-GD3.25-1-UH.avi", def_top_th, def_bot_th))
videos.append(makeObj("CM6-BL6-GD3.5-1-BH.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM12-CD1-DIES-V-NPBS-CON-1-BHP-b.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM1-H2-PGE2-D1-2-a.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM1-H2-SALBUTAMOL-4-b.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM3-H2-OXY-D1.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM5-H1-CONTROL-1.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM5-H1-PGE2D1-2.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM5-H1-PGE2D2-3.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM5-H1-SAL-4.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM5-H2-CONTROL-1.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM5-H2-OXYD1-2.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM7-H1-PGE2D1-2.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM7-H2-CONTROL-1.avi", def_top_th, def_bot_th))
videos.append(makeObj("DM7-H2-OXYD1-2.avi", def_top_th, def_bot_th))
videos.append(makeObj("PM1-H1-OXYTOCIN-D1-2-A.avi", def_top_th, def_bot_th))
videos.append(makeObj("PM1-H1-OXYTOCIN-D1-2-LASXint-A.avi", def_top_th, def_bot_th))
videos.append(makeObj("PM5-H1-PGE2-D1-2.avi", def_top_th, def_bot_th))
videos.append(makeObj("PM5-H1-PGE2-D1-2-LASXint.avi", def_top_th, def_bot_th))
videos.append(makeObj("PM6-H1-CONTROL-1.avi", def_top_th, def_bot_th))
videos.append(makeObj("PM6-H1-OXY-D1-2.avi", def_top_th, def_bot_th))
videos.append(makeObj("SM7-CD1-GD3.25-SO-1-BH.avi", def_top_th, def_bot_th))

print(F'Processing {len(videos)} videos')


# %%
for i, cur_vid in enumerate(videos):
    file_name = cur_vid['file_name'].replace('.avi','')

    checkDirs(join(output_folder, file_name))
    # Copy the video to the output folder
    shutil.copy(join(data_folder,file_name+'.avi'), join(output_folder, file_name,'Original'))

    print(F'******** {file_name} *************')
    pts_per_spline_top = cur_vid['cubic_splines_pts_top']
    pts_per_spline_bottom = cur_vid['cubic_splines_pts_bottom']

    top_edge_th = cur_vid['top_edge_th']
    bottom_edge_th = cur_vid['bottom_edge_th']

    median_filt_size_top = cur_vid['medial_filter_size_top']
    median_filt_size_bottom = cur_vid['medial_filter_size_bottom']

    print('Reading Data....')
    all_video, rows, cols, frames = readFramesFromVideoFile(join(data_folder,file_name+'.avi'))
    # frames = 100  # Just to make it faster for debugging purposes
    # all_video = all_video[0:frames,:,:]
    cubi_spline_pts_top = int((cols/1000) * pts_per_spline_top)
    cubi_spline_pts_bottom = int((cols/1000) * pts_per_spline_bottom)
    print(F'Number of cubic splines points: {cubi_spline_pts_top}')
    print('Done!')

    print('Smoothing Data...')
    # Blurs the image in the X and Time dimensions
    # plotMultipleImages([all_video[0,:,:]], ['Temp'])
    smooth = gaussianBlurXandZ(all_video, k_size_time, k_size_space)
    print('Done!')

    top_pos = np.zeros((frames,cols))
    bottom_pos = np.zeros((frames,cols))
    mean_intensities = np.zeros((frames,cols))
    area_vals = np.zeros((frames,cols))

    print('Computing top and bottom positions....')
    for cur_frame in range(frames):
        # Print the current frame every 100 frames
        if cur_frame % 100 == 0:
            print(F'\t\tFrame: {cur_frame}')

        img = smooth[cur_frame,:,:]
        c_sob = computeEdgesSobel(img, 5)
        # Normalize sobel
        c_sob = c_sob / np.max(c_sob)

        if cur_frame == frame_to_plot:
            # Plots the original image and the smoothed one
            red_colors = LinearSegmentedColormap.from_list('custom_red', [(0, 0, 0), (1, 0, 0)], N=256)
            # Both together
            # plotMultipleImages([all_video[frame_to_plot,:,:], smooth[frame_to_plot,:,:]], ['Original', 'Smoothed'], 
            #                extent=computeExtentSpace(smooth[cur_frame,:,:]), units=['Size in mm', 'Size in mm'],
            #                     cmap=red_colors,
            #                     output_folder=join(output_folder, file_name, 'Steps'),
            #                     file_name=f"1_smooth_{file_name}", auto=False,
            #                 view_results=disp_images)
            # Single one
            plotMultipleImages([smooth[frame_to_plot,:,:]], ['Smoothed'], 
                            extent=computeExtentSpace(smooth[cur_frame,:,:]), units=['Size in mm', 'Size in mm'],
                                cmap=red_colors,
                                output_folder=join(output_folder, file_name, 'Steps'),
                                file_name=f"1_smooth_{file_name}.jpg", auto=False,
                            view_results=disp_images)

            # Both together
            # plotMultipleImages([smooth[cur_frame,:,:], c_sob], ['Smoothed Frame', 'Edge index from Sobel'], view_results=False,
            #                extent=computeExtentSpace(smooth[cur_frame,:,:]), units=['Size in mm', 'Size in mm'],
                                    # output_folder=join(output_folder, file_name, 'Steps'),
            #                     auto=False,
            #                     cmap=cmo.cm.delta,
            #                 file_name = f"2_edgeindex_{file_name}",
            #                cbar_label=['Intensity', 'Edge index from Sobel'], 
            #                draw_vline=int(cols/2))

            # Single one
            plotMultipleImages([c_sob], ['Edge index from Sobel'], view_results=disp_images,
                            extent=computeExtentSpace(smooth[cur_frame,:,:]), units=['Size in mm', 'Size in mm'],
                                output_folder=join(output_folder, file_name, 'Steps'),
                                auto=False,
                                cmap=cmo.cm.delta,
                            file_name = f"2_edgeindex_{file_name}.jpg",
                            cbar_label=['Edge index from Sobel'], 
                            draw_vline=int(cols/2))
            
        # Selects the top position when the 'cumulative' edges overpass a threshold (initial guess)
        top_pos[cur_frame,:] = np.argmax(np.cumsum(c_sob, axis=0) > top_edge_th,axis=0)
        bottom_pos[cur_frame,:] = rows - np.argmax(np.cumsum(np.flip(c_sob,axis=0), axis=0) 
                                                    < -bottom_edge_th, axis=0)

        # ========================= TODO THIS PART IS FOR THE BOTTOM THIRD OF THE HORN =========================
        # Mask just the bottom one third of the detected horn
        top = top_pos[cur_frame,:]
        bot = bottom_pos[cur_frame,:]
        all_range = bot - top
        # top_pos[cur_frame,:] = bot - all_range//3
        top_pos[cur_frame,:] = bot - 50

        if cur_frame == frame_to_plot:
            pos = int(cols/2)
            plt.plot(np.cumsum(c_sob, axis=0)[:, pos], zorder=1)
            # plt.plot(c_sob[:, pos])
            plt.scatter(top_pos[cur_frame,pos]-1, def_top_th, c='r', s=20, zorder=2, label='Selected top position')
            # plt.scatter(c_sob.shape[0] - bottom_pos[cur_frame,pos]-1, -def_bot_th, c='r', s=10, zorder=2, label='Selected bottom position')
            plt.scatter(bottom_pos[cur_frame,pos]-1, def_bot_th+.1, c='k', s=20, zorder=2, label='Selected bottom position')
            plt.xlabel('Row for selected column in the frame')
            plt.ylabel('Cumulative sum')
            plt.title("Cumulative sum of the edge index")
            plt.legend()
            plt.savefig(join(output_folder, file_name, "Steps", f"3_{file_name}.png"), bbox_inches='tight')
            plt.close()
            
        # Plot before computing the median filter
        # if (cur_frame % plot_every_n_frames) == 0:
        #     plotImageAndScatter(all_video[cur_frame, :, :], [top_pos[cur_frame, :], bottom_pos[cur_frame, :]], 
        #                         title=F'Initially detected uterine horn',
        #                         units=['Frame column', 'Frame row'],
        #                         savefig=True, output_folder=join(output_folder, file_name,'MaskArea',file_name+'.jpg'),
        #                         file_name = F'{file_name}_frame_{cur_frame:04d}_BeforeMedian.jpg', view_results=disp_images)

        # ------------ Median filter on the obtained curves --------
        top_pos[cur_frame,:] = medfilt(top_pos[cur_frame], median_filt_size_top)
        bottom_pos[cur_frame,:] = medfilt(bottom_pos[cur_frame], median_filt_size_bottom)

        # For plotting BEFORE the smoothing
        # if (cur_frame % plot_every_n_frames) == 0:
        #     plotImageAndScatter(all_video[cur_frame, :, :], [top_pos[cur_frame, :], bottom_pos[cur_frame, :]], 
        #                         title=F'Median filtered detected uterine horn',
        #                         savefig=True, output_folder=join(output_folder, file_name,'MaskArea',file_name+'.jpg'),
        #                         file_name = F'{file_name}_frame_{cur_frame:04d}_BeforeSCubic.jpg', view_results=disp_images)

        # ------------ Smoothing curve of top positions -------------
        top_pos[cur_frame,:] = cubicSplines(top_pos[cur_frame,:], cubi_spline_pts_top)
        # ------------ Smoothing curve of bottom positions -------------
        bottom_pos[cur_frame,:] = cubicSplines(bottom_pos[cur_frame,:], cubi_spline_pts_bottom)

        if cur_frame == frame_to_plot:
            plotImageAndScatter(all_video[cur_frame, :, :], [top_pos[cur_frame, :], bottom_pos[cur_frame, :]], 
                                title=F'Uterine horn final detection. Time {cur_frame/5:0.1f} s',
                                extent=computeExtentSpace(all_video[cur_frame, :, :]), 
                                units=['Size in mm', 'Size in mm'],
                        savefig=True, 
                        output_folder=join(output_folder, file_name, 'Steps'),
                        file_name = F'4_final_{file_name}.jpg', 
                        view_results=disp_images)


        # Plots the obtained mask and edges, only once every plot_every_n_frames frames
        if (cur_frame % plot_every_n_frames) == 0: # Only plot once every x frames
            # plotImageAndScatter(all_video[cur_frame, :, :], [top_pos[cur_frame, :], bottom_pos[cur_frame, :]], title=F'{file_name} {cur_frame}',
            plotImageAndScatter(all_video[cur_frame, :, :], [top_pos[cur_frame, :], bottom_pos[cur_frame, :]], 
                                title=F'Uterine horn final detection. Time {cur_frame/5:0.1f} s',
                                extent=computeExtentSpace(all_video[cur_frame, :, :]), 
                                units=['Size in mm', 'Size in mm'],
                        savefig=True, output_folder=join(output_folder, file_name,'MaskArea'),
                        file_name = F'{file_name}_frame_{cur_frame:04d}.jpg', view_results=disp_images)

    print('Done!')

    bottom_pos = bottom_pos.astype(int)
    top_pos = top_pos.astype(int)

    # ========================================================================================================
    print('Computing means intensities and areas!')
    for cur_frame in range(frames):
        mask = np.zeros((rows,cols))
        for cur_col in range(cols):
            mask[top_pos[cur_frame,cur_col]:bottom_pos[cur_frame,cur_col],cur_col] =  1


        mean_intensities[cur_frame,:] = np.true_divide(all_video[cur_frame,:,:].sum(0), (mask != False).sum(0))
        # if (cur_frame % plot_every_n_frames) == 0: # Only plot once every x frames
        #     plotImageAndMask(all_video[cur_frame,:,:],mean_intensities[cur_frame,:],savefig=True,
        #                         output_folder=join(output_folder, file_name,'Mask_Area'),
        #                         file_name = F'{file_name}_Mask_frame_{cur_frame:04d}.jpg')

    area_vals = bottom_pos - top_pos
    print('Done!')

    print('Saving results...')
    final_folder = join(output_folder, file_name,'Original')
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