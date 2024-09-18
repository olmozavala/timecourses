# %%
from os.path import join
import shutil
import os
import cv2
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.signal import medfilt, argrelextrema 
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from FiltersAndUtils import *
from Utils_Visualization import *
from Utils_io import *
from Utils_Viz import *
import time


def process_video(video_name, cur_data_folder, cur_output_folder, display=False, create_animation=False, th_top=0.2, th_bottom=0.2):

    checkDirs(cur_output_folder)

    sample_frame = 0
    col_to_plot = 590
    file_name = video_name + '.avi'
    steps_folder = join(cur_output_folder, 'Original')

    # Copy the video to the output folder
    if not os.path.exists(join(cur_output_folder, file_name)):
        shutil.copy(join(cur_data_folder,file_name), join(cur_output_folder, file_name))

    print(F'******** {file_name} *************')

    # %%
    print('Reading Data....')
    start_time = time.time()
    all_video, rows, cols, frames = readFramesFromVideoFile(join(cur_data_folder,file_name))
    end_time = time.time()
    print(f'Video reading time: {end_time - start_time:.2f} seconds')
    plot_frame(all_video, sample_frame, title='Original', output_folder=steps_folder)

    # %%
    # Normalize each column in each frame of the video
    print('Normalizing video data...')
    start_time = time.time()
    # Compute mean and max for each column in each frame
    column_means = np.mean(all_video, axis=1, keepdims=True)
    column_maxs = np.max(all_video, axis=1, keepdims=True)
    # Normalize the entire video at once
    normalized_video = (all_video - column_means) / (column_maxs - column_means)
    # Convert to float32 to save memory
    normalized_video = normalized_video.astype(np.float32)
    end_time = time.time()
    print(f'Normalization time: {end_time - start_time:.2f} seconds')

    # Plot a sample frame to verify normalization
    plot_frame(normalized_video, sample_frame, title='Normalized', output_folder=steps_folder)

    # Plot the intensity curve for a specific column
    plot_curve(normalized_video, sample_frame, col_to_plot, title='Intensity Curve', output_folder=steps_folder)

    # %% Identify the top edges by column as the row where the intensity is greater than 0.5
    print('Identifying top edges...')
    start_time = time.time()
    top_edges = np.argmax(normalized_video > th_top, axis=1)
    top_edges_time = time.time() - start_time
    print(f'Top edges identification time: {top_edges_time:.2f} seconds')
    # %%
    print('Identifying bottom edges...') # In this case we need to look in 'reversed' order from the bottom
    start_time = time.time()
    bottom_edges = rows - np.argmax(normalized_video[:,::-1,:] > th_bottom, axis=1)
    bottom_edges_time = time.time() - start_time
    print(f'Bottom edges identification time: {bottom_edges_time:.2f} seconds')
    print(f'Total edge identification time: {top_edges_time + bottom_edges_time:.2f} seconds')

    # %% Smooth the edges
    print('Smoothing edges...')
    start_time = time.time()
    # Remove single outliers
    top_edges = medfilt(top_edges, kernel_size=(1, 3))
    bottom_edges = medfilt(bottom_edges, kernel_size=(1, 3))
    # Remove continuous outliers
    outlier_th = 40
    for i in range(0, frames):
        for col in range(1, cols-1):
            if abs(top_edges[i,col] - top_edges[i,col-1]) > outlier_th:
                top_edges[i,col] = top_edges[i,col-1] 
            if abs(bottom_edges[i,col] - bottom_edges[i,col-1]) > outlier_th:
                bottom_edges[i,col] = bottom_edges[i,col-1] 

    sigma = 3
    top_edges = gaussian_filter1d(top_edges, sigma=sigma, axis=1)
    bottom_edges = gaussian_filter1d(bottom_edges, sigma=sigma, axis=1)
    # Fill the edges with the last value
    top_edges[:, 0] = top_edges[:, 1]  # Fill left border
    top_edges[:, -1] = top_edges[:, -2]  # Fill right border
    bottom_edges[:, 0] = bottom_edges[:, 1]  # Fill left border
    bottom_edges[:, -1] = bottom_edges[:, -2]  # Fill right border

    end_time = time.time()
    print(f'Edges smoothing time: {end_time - start_time:.2f} seconds')

    # %% Computing area and mean intensity from smoothed edges
    print('Computing area ...')
    area = (bottom_edges - top_edges) * cols
    print('Done!')
    
    # %%
    print('Computing mean intensity ...')
    mean_intensity = np.zeros_like(area)
    # for i in range(frames):
    #     for col in range(cols):
    #         mean_intensity[i,col] = np.mean(all_video[i, top_edges[i,col]:bottom_edges[i,col], col])
    # print('Done!')

    # Plot the top and bottom edges as an image
    plot_fields(video_name, top_edges, title='Top positions', display=display, output_folder=steps_folder)
    # Plot the bottom edges as an image
    plot_fields(video_name, bottom_edges, title='Bottom positions', display=display, output_folder=steps_folder)
    # Plot the area as an image
    plot_fields(video_name, area, title='Area', display=display, output_folder=steps_folder)
    # Plot the mean intensity as an image
    plot_fields(video_name, mean_intensity, title='Mean intensities', display=display, output_folder=steps_folder)

    # Save each of them as csv files
    np.savetxt(join(steps_folder, f'{video_name}_Top_edges.csv'), top_edges.astype(int), delimiter=',', fmt='%d')
    np.savetxt(join(steps_folder, f'{video_name}_Bottom_edges.csv'), bottom_edges.astype(int), delimiter=',', fmt='%d')
    np.savetxt(join(steps_folder, f'{video_name}_Area.csv'), area.astype(int), delimiter=',', fmt='%d')
    np.savetxt(join(steps_folder, f'{video_name}_Mean_intensity.csv'), mean_intensity.astype(int), delimiter=',', fmt='%d')

    # %% Make video that shows the top and bottom edges for the first 100 frames
    # Create a figure and axis
    if create_animation:
        print('Creating animation...')
        start_time = time.time()
        generate_animation(all_video, top_edges, bottom_edges, steps_folder, offset=600)
        end_time = time.time()
        print(f'Animation creation time: {end_time - start_time:.2f} seconds')

        print(f'Done for {video_name}')

# %% ----------------------------------- MAIN ---------------------------------

videos_path = 'SpecificWaves'
data_folder = F'/data/RiplaUterusWaves/Videos/{videos_path}'
root_output_folder = F'/data/RiplaUterusWaves/Outputs/{videos_path}OneThirdBottomIntensities'
disp_images = False # Indicates if we want to see the images as they are processed

videos = []
videos.append({"name":"DCCM3-DIES-CD1-CON-1-BH-haftest","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"BPM1-H1-PGE2D1-2","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"BPM2-H2-PGE2D1-2","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"BPM3-H1-CONTROL-1","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"CM3-BL6-GD3.25-1-UH","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"CM6-BL6-GD3.5-1-BH","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM12-CD1-DIES-V-NPBS-CON-1-BHP-b","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM1-H2-PGE2-D1-2-a","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM1-H2-SALBUTAMOL-4-b","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM3-H2-OXY-D1","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM5-H1-CONTROL-1","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM5-H1-PGE2D1-2","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM5-H1-PGE2D2-3","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM5-H1-SAL-4","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM5-H2-CONTROL-1","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM5-H2-OXYD1-2","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM7-H1-PGE2D1-2","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM7-H2-CONTROL-1","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"DM7-H2-OXYD1-2","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"PM1-H1-OXYTOCIN-D1-2-A","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"PM1-H1-OXYTOCIN-D1-2-LASXint-A","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"PM5-H1-PGE2-D1-2","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"PM5-H1-PGE2-D1-2-LASXint","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"PM6-H1-CONTROL-1","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"PM6-H1-OXY-D1-2","th_top":0.2,"th_bottom":0.2})
# videos.append({"name":"SM7-CD1-GD3.25-SO-1-BH","th_top":0.2,"th_bottom":0.2})
print(F'Processing {len(videos)} videos')

# %%
for cur_video in videos:
    name = cur_video["name"]
    output_folder = join(root_output_folder, name)
    process_video(name, data_folder, output_folder, display=disp_images, create_animation=True, th_top=cur_video["th_top"], th_bottom=cur_video["th_bottom"])