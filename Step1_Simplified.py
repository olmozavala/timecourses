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
from FiltersAndUtils import *
from Utils_Visualization import *
from Utils_io import *
import time

#%%
def plot_frame(video, frame_idx, title=''):
    plt.figure()
    plt.imshow(video[frame_idx,:,:], cmap='Reds')
    plt.title(f'{title} Frame {frame_idx}')
    plt.axis('off')
    plt.show()

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

plot_every_n_frames = 40

videos = []
videos.append("BPM1-H1-PGE2D1-2")
# videos.append(join(data_folder,"BPM2-H2-PGE2D1-2.avi"))
# videos.append(join(data_folder,"BPM3-H1-CONTROL-1.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"CM3-BL6-GD3.25-1-UH.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"CM6-BL6-GD3.5-1-BH.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM12-CD1-DIES-V-NPBS-CON-1-BHP-b.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM1-H2-PGE2-D1-2-a.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM1-H2-SALBUTAMOL-4-b.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM3-H2-OXY-D1.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM5-H1-CONTROL-1.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM5-H1-PGE2D1-2.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM5-H1-PGE2D2-3.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM5-H1-SAL-4.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM5-H2-CONTROL-1.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM5-H2-OXYD1-2.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM7-H1-PGE2D1-2.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM7-H2-CONTROL-1.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"DM7-H2-OXYD1-2.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"PM1-H1-OXYTOCIN-D1-2-A.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"PM1-H1-OXYTOCIN-D1-2-LASXint-A.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"PM5-H1-PGE2-D1-2.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"PM5-H1-PGE2-D1-2-LASXint.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"PM6-H1-CONTROL-1.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"PM6-H1-OXY-D1-2.avi", def_top_th, def_bot_th))
# videos.append(join(data_folder,"SM7-CD1-GD3.25-SO-1-BH.avi", def_top_th, def_bot_th))
print(F'Processing {len(videos)} videos')


# %%
cur_video = videos[0]
file_name = join(data_folder,cur_video+'.avi')

# Copy the video to the output folder
if not os.path.exists(join(output_folder, file_name)):
    shutil.copy(join(data_folder,file_name), join(output_folder, file_name))

print(F'******** {file_name} *************')

# %%
print('Reading Data....')
start_time = time.time()
all_video, rows, cols, frames = readFramesFromVideoFile(join(data_folder,file_name))
end_time = time.time()
print(f'Video reading time: {end_time - start_time:.2f} seconds')

idx = 0
plot_frame(all_video, idx, title='Original')

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

# Replace all_video with normalized_video for further processing
normalized_video

# Plot a sample frame to verify normalization
sample_frame = 0
plot_frame(normalized_video, sample_frame, title='Normalized')

# %%
# Plot the curve of intensities for a given column and frame
col_to_plot = 590
frame_to_plot = 10

# Figure 1: Frame with column location
plt.figure(figsize=(10, 5))
plt.imshow(normalized_video[frame_to_plot,:,:], cmap='gray')
plt.axvline(x=col_to_plot, color='r', linestyle='--')
plt.title(f'Frame {frame_to_plot} with column {col_to_plot} highlighted')
plt.colorbar(label='Intensity')
plt.show()

# Figure 2: Intensity curve for the selected column
plt.figure(figsize=(10, 5))
plt.plot(normalized_video[frame_to_plot, :, col_to_plot])
plt.title(f'Intensity curve for column {col_to_plot} at frame {frame_to_plot}')
plt.xlabel('Row')
plt.ylabel('Intensity')
plt.show()

# %% Identify the top edges by column as the row where the intensity is greater than 0.5
print('Identifying top edges...')
start_time = time.time()
top_edges = np.argmax(normalized_video > 0.2, axis=1)
top_edges_time = time.time() - start_time
print(f'Top edges identification time: {top_edges_time:.2f} seconds')
# %%
print('Identifying bottom edges...') # In this case we need to look in 'reversed' order from the bottom
start_time = time.time()
bottom_edges = rows - np.argmax(normalized_video[:,::-1,:] > 0.2, axis=1)
bottom_edges_time = time.time() - start_time
print(f'Bottom edges identification time: {bottom_edges_time:.2f} seconds')
print(f'Total edge identification time: {top_edges_time + bottom_edges_time:.2f} seconds')

# Plot the top as scattered dots on top of the frame
# %%
frame_to_plot = 110
plt.figure(figsize=(10, 5))
plt.imshow(normalized_video[frame_to_plot,:,:], cmap='gray')
plt.scatter(range(cols), top_edges[frame_to_plot,:], color='red', s=1)
plt.scatter(range(cols), bottom_edges[frame_to_plot,:], color='blue', s=1)
plt.title(f'Top edges for column at frame {frame_to_plot}')
plt.xlabel('Column')
plt.ylabel('Row')
plt.show()

# %% Smooth the edges
top_edges = medfilt(top_edges, kernel_size=5)
bottom_edges = medfilt(bottom_edges, kernel_size=5)

# %% Make video that shows the top and bottom edges for the first 100 frames

# Create a figure and axis
fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

# Function to update the plot for each frame
def update(frame_idx):
    if frame_idx % 10 == 0:
        print(f'Processing frame {frame_idx}')
    ax.clear()
    ax.imshow(normalized_video[frame_idx,:,:], cmap='gray', aspect='auto')
    ax.scatter(range(cols), top_edges[frame_idx,:], color='red', s=2, alpha=0.7)
    ax.scatter(range(cols), bottom_edges[frame_idx,:], color='blue', s=2, alpha=0.7)
    ax.set_title(f'Top and bottom edges for frame {frame_idx}', fontsize=14)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=200, interval=50)

# Save the animation as an MP4 file with higher quality
writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=5000)
anim.save('edges_animation_high_quality.mp4', writer=writer)

plt.close(fig)
# %%
