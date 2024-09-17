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
def plot_frame(video, frame_idx, title='', display=False, output_folder=''):
    plt.figure()
    plt.imshow(video[frame_idx,:,:], cmap='Reds')
    plt.title(f'{title} Frame {frame_idx}', fontsize=14)
    plt.axis('off')
    if display:
        plt.show()
    else:
        plt.savefig(f'{title}_Frame_{frame_idx}.png')
        plt.close()

def plot_fields(video_name, field, title='', display=False, output_folder=''):
    plt.figure(figsize=(16, 9), dpi=120)
    plt.imshow(field[::-1], cmap='gray', origin='lower')
    plt.colorbar(label=title)
    plt.title(f'{title}', fontsize=16)
    plt.xlabel('Column', fontsize=14) 
    plt.ylabel('Frame', fontsize=14)
    if display:
        plt.show()
    else:
        plt.savefig(join(output_folder, f'{video_name}_{title.replace(" ", "_")}.jpg'))
        plt.close()

def checkDirs(output_folder):
    saveDir(join(output_folder, 'MaskArea'))
    saveDir(join(output_folder, 'Curves'))
    saveDir(join(output_folder, 'FilteredImages'))
    saveDir(join(output_folder, 'Original'))


def process_video(video_name, cur_data_folder, cur_output_folder, display=False):

    checkDirs(cur_output_folder)
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

    idx = 0
    plot_frame(all_video, idx, title='Original', output_folder=steps_folder)

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
    plot_frame(normalized_video, sample_frame, title='Normalized', output_folder=steps_folder)

    # %%
    # Plot the curve of intensities for a given column and frame
    col_to_plot = 590
    frame_to_plot = 10

    # Figure 1: Frame with column location
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    # Figure 1: Frame with column location
    im = ax1.imshow(normalized_video[frame_to_plot,:,:], cmap='gray')
    ax1.axvline(x=col_to_plot, color='r', linestyle='--')
    ax1.set_title(f'Frame {frame_to_plot} with column {col_to_plot} highlighted')
    plt.colorbar(im, ax=ax1, label='Intensity')

    # Figure 2: Intensity curve for the selected column
    ax2.plot(normalized_video[frame_to_plot, :, col_to_plot])
    ax2.set_title(f'Intensity curve for column {col_to_plot} at frame {frame_to_plot}')
    ax2.set_xlabel('Row')
    ax2.set_ylabel('Intensity')

    plt.tight_layout()
    if display:
        plt.show()
    else:
        plt.savefig(join(steps_folder, f'Intensity curve Frame_{frame_to_plot} Column_{col_to_plot}.png'))
        plt.close()

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

    # %% Smooth the edges
    print('Smoothing edges...')
    start_time = time.time()
    top_edges = medfilt(top_edges, kernel_size=(1, 5))
    bottom_edges = medfilt(bottom_edges, kernel_size=(1, 5))

    # Fill the borders with the last value
    top_edges[:, 0] = top_edges[:, 1]  # Fill left border
    top_edges[:, -1] = top_edges[:, -2]  # Fill right border
    bottom_edges[:, 0] = bottom_edges[:, 1]  # Fill left border
    bottom_edges[:, -1] = bottom_edges[:, -2]  # Fill right border

    end_time = time.time()
    print(f'Edge smoothing time: {end_time - start_time:.2f} seconds')

    # %% Computing area and mean intensity from smoothed edges
    area = (bottom_edges - top_edges) * cols
    
    mean_intensity = np.zeros_like(area)
    print(f"Computing mean intensity ...")
    for i in range(frames):
        for col in range(cols):
            mean_intensity[i,col] = np.mean(all_video[i, top_edges[i,col]:bottom_edges[i,col], col], axis=0)
    print("Done!")
    # Plot the top as scattered dots on top of the frame
    # %%
    frame_to_plot = 110
    plt.figure(figsize=(10, 5))
    plt.imshow(normalized_video[frame_to_plot,:,:], cmap='gray')
    plt.scatter(range(cols), top_edges[frame_to_plot,:], color='red', s=1)
    plt.scatter(range(cols), bottom_edges[frame_to_plot,:], color='blue', s=1)
    plt.title(f'Top & Bottom edges for frame {frame_to_plot}')
    plt.xlabel('Column')
    plt.ylabel('Row')
    if display:
        plt.show()
    else:
        plt.savefig(join(steps_folder, f'TopBottom_edges _rame_{frame_to_plot}.jpg'))
        plt.close()

    # %% Make video that shows the top and bottom edges for the first 100 frames
    # Create a figure and axis
    print('Creating animation...')
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

    # Function to update the plot for each frame
    def update(frame_idx):
        if frame_idx % 10 == 0:
            print(f'Processing frame {frame_idx}')
        ax.clear()
        ax.imshow(all_video[frame_idx,:,:], cmap='gray', aspect='auto')
        ax.scatter(range(cols), top_edges[frame_idx,:], color='red', s=2, alpha=0.7)
        ax.scatter(range(cols), bottom_edges[frame_idx,:], color='blue', s=2, alpha=0.7)
        ax.set_title(f'Top and bottom edges for frame {frame_idx}', fontsize=14)
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=100, interval=50)

    # Save the animation as an MP4 file with higher quality
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=5000)
    anim.save(join(steps_folder, f'edges_animation_high_quality.mp4'), writer=writer)

    plt.close(fig)
    end_time = time.time()
    print(f'Animation creation time: {end_time - start_time:.2f} seconds')

    print(f'Done for {video_name}')

    # Plot the top and bottom edges as an image
    plot_fields(video_name, top_edges, title='Top positions', display=display, output_folder=steps_folder)
    # Plot the bottom edges as an image
    plot_fields(video_name, bottom_edges, title='Bottom positions', display=display, output_folder=steps_folder)
    # Plot the area as an image
    plot_fields(video_name, area, title='Area', display=display, output_folder=steps_folder)
    # Plot the mean intensity as an image
    plot_fields(video_name, mean_intensity, title='Mean intensities', display=display, output_folder=steps_folder)
    
# %%

# %% ----------------------------------- MAIN ---------------------------------

videos_path = 'SpecificWaves'
data_folder = F'/data/RiplaUterusWaves/Videos/{videos_path}'
root_output_folder = F'/data/RiplaUterusWaves/Outputs/{videos_path}OneThirdBottomIntensities'
disp_images = False # Indicates if we want to see the images as they are processed

videos = []
videos.append("BPM1-H1-PGE2D1-2")
videos.append("BPM2-H2-PGE2D1-2")
videos.append("BPM3-H1-CONTROL-1")
videos.append("CM3-BL6-GD3.25-1-UH")
videos.append("CM6-BL6-GD3.5-1-BH")
videos.append("DM12-CD1-DIES-V-NPBS-CON-1-BHP-b")
videos.append("DM1-H2-PGE2-D1-2-a")
videos.append("DM1-H2-SALBUTAMOL-4-b")
videos.append("DM3-H2-OXY-D1")
videos.append("DM5-H1-CONTROL-1")
videos.append("DM5-H1-PGE2D1-2")
videos.append("DM5-H1-PGE2D2-3")
videos.append("DM5-H1-SAL-4")
videos.append("DM5-H2-CONTROL-1")
videos.append("DM5-H2-OXYD1-2")
videos.append("DM7-H1-PGE2D1-2")
videos.append("DM7-H2-CONTROL-1")
videos.append("DM7-H2-OXYD1-2")
videos.append("PM1-H1-OXYTOCIN-D1-2-A")
videos.append("PM1-H1-OXYTOCIN-D1-2-LASXint-A")
videos.append("PM5-H1-PGE2-D1-2")
videos.append("PM5-H1-PGE2-D1-2-LASXint")
videos.append("PM6-H1-CONTROL-1")
videos.append("PM6-H1-OXY-D1-2")
videos.append("SM7-CD1-GD3.25-SO-1-BH")
print(F'Processing {len(videos)} videos')

# %%
for cur_video in videos:
    output_folder = join(root_output_folder, cur_video)
    process_video(cur_video, data_folder, output_folder, display=disp_images)