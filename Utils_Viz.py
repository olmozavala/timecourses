import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os.path import join

def save_fig(display, file_name, output_folder):
    if display:
        plt.show()
    else:
        plt.savefig(join(output_folder, file_name))
        plt.close()

#%%
def plot_frame(video, frame_idx, title='', display=False, output_folder=''):
    plt.figure()
    plt.imshow(video[frame_idx,:,:], cmap='Reds')
    plt.title(f'{title} Frame {frame_idx}', fontsize=14)
    plt.axis('off')
    save_fig(display, f'{title}_Frame_{frame_idx}.png', output_folder)

def plot_fields(video_name, field, title='', display=False, output_folder=''):
    """
    Plot a 2D field as an image.

    Args:
        video_name (str): Name of the video, used in the output filename.
        field (numpy.ndarray): 2D array representing the field to plot.
        title (str, optional): Title for the plot. Defaults to ''.
        display (bool, optional): If True, display the plot. If False, save it. Defaults to False.
        output_folder (str, optional): Folder to save the plot if not displayed. Defaults to ''.

    Returns:
        None

    This function creates a figure, plots the given field as an image,
    adds a colorbar, sets labels and title, and either displays or saves the plot.
    """
    plt.figure(figsize=(16, 9), dpi=120)
    plt.imshow(field, cmap='gray', origin='lower')
    plt.colorbar(label=title)
    plt.title(f'{title}', fontsize=16)
    plt.xlabel('Column', fontsize=14) 
    plt.ylabel('Frame', fontsize=14)
    save_fig(display, f'{video_name}_{title.replace(" ", "_")}.jpg', output_folder)

def plot_curve(normalized_video, frame_to_plot, col_to_plot, title='', display=False, output_folder=''):

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
    save_fig(display, f'Intensity curve Frame_{frame_to_plot} Column_{col_to_plot}.png', output_folder)

def generate_animation(all_video, top_edges, bottom_edges, steps_folder, num_frames=200, offset=0):
    cols = all_video.shape[2]
    fig, ax = plt.subplots(figsize=(20, 3), dpi=150)

    # Function to update the plot for each frame
    def update(frame_idx):
        frame_idx += offset
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
    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)

    # Save the animation as an MP4 file with higher quality
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=5000)
    anim.save(join(steps_folder, f'edges_animation_high_quality.mp4'), writer=writer)

    plt.close(fig)