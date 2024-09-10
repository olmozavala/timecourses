# %%
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from FiltersAndUtils import *
from Utils_io import *
from Utils_Visualization import *
from scipy.ndimage import gaussian_filter
import cv2

def makeObj(name):
    vid = {
        'file_name':name, # Name of the file
    }
    return vid


def bandPassFilter(df):
    f = 5 # 5 Frames por segundo
    pos = 410
    # data = df.iloc[:,pos].values[200:700]
    data = df.iloc[:,pos].values
    nsamples = data.shape[0]
    T = nsamples/f # Total seconds
    t = np.linspace(0,T,nsamples)
    lowcut = 1/20 # Remove anything slower than  20 seconds
    highcut = 1/5  # Remove anything faster than 5 seconds

    plt.plot(t, data, label='Noisy signal')
    plt.title(cur_f_type)
    # plt.show()
    y = butter_bandpass_filter(data, lowcut, highcut, f, order=6)
    plt.plot(t, y, label=F'Filtered signals between {1/lowcut} fps and {1/highcut} fps')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.title(F'{cur_vid} {cur_f_type} filtered {pos}')
    plt.legend(loc='best')
    plt.ylim(-2,2)
    # plt.show()

# This is the second file that needs to be executed. It subtracts the low frequencies and removes high frecuencies.
input_folder = '/data/RiplaUterusWaves/Outputs/SpecificWaves'
videos = []
plot_every_n_frames = 10
disp_images = False # Indicates if we want to see the images as they are processed

videos = []
# videos.append(makeObj("BPM1-H1-PGE2D1-2"))
# videos.append(makeObj("BPM2-H2-PGE2D1-2"))
# videos.append(makeObj("BPM3-H1-CONTROL-1"))
# videos.append(makeObj("CM3-BL6-GD3.25-1-UH"))
# videos.append(makeObj("CM6-BL6-GD3.5-1-BH"))
# videos.append(makeObj("DM12-CD1-DIES-V-NPBS-CON-1-BHP-b"))
# videos.append(makeObj("DM1-H2-PGE2-D1-2-a"))
# videos.append(makeObj("DM1-H2-SALBUTAMOL-4-b"))
# videos.append(makeObj("DM3-H2-OXY-D1"))
# videos.append(makeObj("DM5-H1-CONTROL-1"))
# videos.append(makeObj("DM5-H1-PGE2D1-2"))
# videos.append(makeObj("DM5-H1-PGE2D2-3"))
# videos.append(makeObj("DM5-H1-SAL-4"))
# videos.append(makeObj("DM5-H2-CONTROL-1"))
# videos.append(makeObj("DM5-H2-OXYD1-2"))
# videos.append(makeObj("DM7-H1-PGE2D1-2"))
# videos.append(makeObj("DM7-H2-CONTROL-1"))
# videos.append(makeObj("DM7-H2-OXYD1-2"))
# videos.append(makeObj("PM1-H1-OXYTOCIN-D1-2-A"))
# videos.append(makeObj("PM1-H1-OXYTOCIN-D1-2-LASXint-A"))
# videos.append(makeObj("PM5-H1-PGE2-D1-2"))
videos.append(makeObj("PM5-H1-PGE2-D1-2-LASXint"))
# videos.append(makeObj("PM6-H1-CONTROL-1"))
# videos.append(makeObj("PM6-H1-OXY-D1-2"))
# videos.append(makeObj("SM7-CD1-GD3.25-SO-1-BH"))

#f_types = ['Area', 'Bottom_positions', 'Mean_intensities', 'Top_positions']
# f_types = ['Area', 'Bottom_positions', 'Mean_intensities']
f_types = ['Mean_intensities', 'Area']

for i, cur_vid in enumerate(videos):
    file_name = cur_vid['file_name']
    data_folder = F'{input_folder}/{file_name}/Original/'
    output_folder = F'{input_folder}/{file_name}/FilteredImages/'
    saveDir(output_folder)

    # Iterates over all the types of data we want to work with 
    for j, cur_f_type in enumerate(f_types):
        try:
            df = pd.read_csv(join(data_folder,file_name+'_'+cur_f_type+'.csv'), dtype=np.float32)
            rows,cols = df.shape

            original_values = df.values
            print(F'Working with file {file_name} filter {cur_f_type}')

            print(F'Saving original data...')
            # Save as JPG
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Original')
            title=F'{file_name}  {cur_f_type} Original'
            # original_norm = (original_values - np.amin(original_values))/np.ptp(original_values)
            # plotFinalFigures(original_values, title, new_file_name+'.jpg',computeExtent(original_values))
            plotFinalFigures(original_values, title, new_file_name+'.jpg',[], view_results=disp_images)

            # Save original data as HTML
            # title=F'{file_name} {cur_f_type}'
            # html_file_name = F'{new_file_name}.html'
            # plotHeatmatPlotty(original_values, rows, cols, title, html_file_name)

            # k = 21
            k = 5
            # Save as Edges
            title=F'{file_name} {cur_f_type} horizontal edge '
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Horizontal_Edge_Original')
            edge_h = cv2.Sobel(original_values,cv2.CV_64F,0,1,ksize=k)
            edge_h = edge_h.clip(min=-50, max=50)
            # plotFinalFigures(edge, title, new_file_name+'.jpg',computeExtent(edge), view_results=disp_images)
            plotFinalFigures(edge_h, title, new_file_name+'.jpg',[], view_results=disp_images)
            # This is an example on how to run it specifyin gthe zmin and zmax
            plotHeatmatPlotty(edge_h, rows, cols, title, new_file_name+'.html', zmin=-50, zmax=50)
            # np.savetxt(F'{new_file_name}.csv', edge,fmt='%10.3f', delimiter=',')

            # k = 31
            k = 5
            title=F'{file_name} {cur_f_type} vertical edge original'
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Vertical_Edge')
            edge_v = cv2.Sobel(original_values,cv2.CV_64F, 1, 0,ksize=k)
            edge_v = (edge_v - np.amin(edge_v))/np.ptp(edge_v)
            # plotFinalFigures(edge, title, new_file_name+'.jpg',computeExtent(edge))
            plotFinalFigures(edge_v, title, new_file_name+'.jpg',[], view_results=disp_images)
            plotHeatmatPlotty(edge_v, rows, cols, title, new_file_name+'.html')
            np.savetxt(F'{new_file_name}.csv', edge_v,fmt='%10.3f', delimiter=',')

            print(F'Done!')

            # bandPassFilter(df) # If we want to properly do a bandpass filter
            print(F'Filtering the data...')
            clean_intensities = np.zeros((rows,cols))
            for frame in df.index.values:
                intensities = original_values[frame,:]
                low_freq = smoothSingleCurve(intensities, 40) # Gets low frequencies
                removed_low = intensities - low_freq
                clean_intensities[frame,:] = smoothSingleCurve(removed_low, 3) # Removes high frequencies
                # clean_intensities[frame,:]= intensities
                # clean_intensities[frame,:]= removed_low
                # clean_intensities[frame,:] = smoothSingleCurve(intensities, 4) # Removes high frequencies
            print(F'Done!')

            print(F'Saving filtered data...')
            # Save as JPG
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Filtered')
            title=F'{file_name}  Filtered'
            # plotFinalFigures(clean_intensities, title, new_file_name+'.jpg',computeExtentSpaceTime(clean_intensities), view_results=disp_images)

            # def plotMultipleImages(imgs, titles=[], output_folder='', file_name='', view_results=True):
            # plotMultipleImages([original_values, edge_h, edge_v, clean_intensities], 
            # Normalize columns
            # clean_intensities = (clean_intensities - np.amin(clean_intensities, axis=0))/np.ptp(clean_intensities)
            # Normalize rows
            # min_by_row = np.amin(clean_intensities, axis=1)
            # max_by_row = np.amax(clean_intensities, axis=1)
            # clean_intensities = (clean_intensities - min_by_row[:,None])/(max_by_row[:,None]-min_by_row[:,None])
            # Mean absolute difference by row
            # Difference between adjacent rows
            # clean_intensities = np.diff(clean_intensities, axis=0)
            plotMultipleImages([original_values, edge_v, clean_intensities], 
                                [F'{cur_f_type} Original', f'{cur_f_type} Vertical Edge', f'{cur_f_type} Filtered Vertical Edge'], 
                                output_folder, F'{file_name}_{cur_f_type}_ALL.jpg', 
                            #    cbar_label=['Original', 'Vertical Edge', 'Filtered Vertical Edge'],
                                extent=computeExtentSpaceTime(original_values),
                                units=['Millimiters','Seconds'],
                                view_results=disp_images, flip=True)

            # 3D Surface plot
            # limits = computeExtentSpaceTime(original_values)
            # X, Y = np.mgrid[limits[2]:limits[3]:rows*1j, limits[0]:limits[1]:cols*1j]

            # fig = plt.figure(figsize=(22, 7))
            # ax = fig.add_subplot(111, projection='3d')
            # Smooth the data with gaussian filter
            # smooth_clean = gaussian_filter(clean_intensities, sigma=10)
            # surf = ax.plot_surface(X, Y, smooth_clean, cmap=cmo.cm.thermal, linewidth=0, antialiased=False)
            # Setting x and y axis labels
            # ax.set_xlabel('Seconds')
            # ax.set_ylabel('Millimiters')
            # Set z axis limits
            # ax.set_zlim(-4, 4)
            # Equal aspect ratio
            # plt.savefig( F'{file_name}_{cur_f_type}_ALL_3D.jpg', bbox_inches='tight')
            # plt.show()

            # Mayavi
            # Create the plot
            # from mayavi import mlab
            # X, Y = np.mgrid[limits[2]:limits[3]:rows*1j, limits[0]:limits[1]*8:cols*1j]
            # mlab.figure(bgcolor=(0, 0, 0))  # Set the background color to white
            # smooth_clean = gaussian_filter(clean_intensities, sigma=10)
            # surf = mlab.surf(X, Y, smooth_clean, colormap='plasma')  # Use the 'coolwarm' colormap
            # mlab.view(azimuth=45, elevation=75, distance=10)  # Set a good viewing angle
            # # Adding axes
            # mlab.title('My 3D Plot')
            # axes = mlab.axes(surf, xlabel='Seconds', ylabel='Millimiters', zlabel='Intensity')
            # axes.label_text_property.color = (0, 0, 0)  # Red color
            # mlab.draw()
            # mlab.show()

            # Save as CSV
            np.savetxt(F'{new_file_name}.csv', clean_intensities,fmt='%10.3f', delimiter=',')

            # ------------------- Save as HTML
            title=F'{file_name} {cur_f_type}'
            html_file_name = F'{new_file_name}.html'
            plotHeatmatPlotty(clean_intensities, rows, cols, title, html_file_name)

            title=F'{file_name} {cur_f_type} horizontal edge filtered'
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Horizontal_Edge_Filtered')
            plotFinalFigures(edge_h, title, new_file_name+'.jpg',computeExtentSpaceTime(edge_h), view_results=disp_images)
            plotHeatmatPlotty(edge_h, rows, cols, title, new_file_name+'.html')

            title=F'{file_name} {cur_f_type} vertical edge filtered'
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Vertical_Edge_Filtered')
            plotFinalFigures(edge_v, title, new_file_name+'.jpg',computeExtentSpaceTime(edge_v), view_results=disp_images)
            plotHeatmatPlotty(edge_v, rows, cols, title, new_file_name+'.html', zmin=0.2, zmax=0.7)
            plotHeatmatPlotty(edge_v, rows, cols, title, new_file_name+'_3D.html', surface=True)

            # Save as CSV
            np.savetxt(F'{new_file_name}.csv', edge_v,fmt='%10.3f', delimiter=',')
        except Exception as e:
            print(F'Error processing {file_name} {cur_f_type}')
            print(e)
            continue
print("Done!!!")
