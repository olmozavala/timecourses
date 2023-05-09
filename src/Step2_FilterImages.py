from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from FiltersAndUtils import *
from Utils_io import *
from Utils_Visualization import *
import cv2

def makeObj(name):
    vid = {
        'file_name':name, # Name of the file
    }
    return vid

def computeExtent(img):
    fps = 2
    pix_per_mil = 56

    frames, pixels = img.shape
    miny = 0
    maxy = frames/fps

    minx = 0
    maxx = pixels/pix_per_mil

    return [minx,maxx,maxy,miny]

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
    plt.show()
    y = butter_bandpass_filter(data, lowcut, highcut, f, order=6)
    plt.plot(t, y, label=F'Filtered signals between {1/lowcut} fps and {1/highcut} fps')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.title(F'{cur_vid} {cur_f_type} filtered {pos}')
    plt.legend(loc='best')
    plt.ylim(-2,2)
    plt.show()

if __name__ == '__main__':
    # This is the second file that needs to be executed. It subtracts the low frequencies and removes high frecuencies.

    gd_video = 'GD4'
    input_folder = '/data/DianaVideosOutput'
    videos = []
    plot_every_n_frames = 10

    if gd_video == 'GD3':
        # ================= GD3 ==================
        # videos.append(makeObj('RGD3T4M01H01'))
        # videos.append(makeObj('RDG3T4M01H01Sal1'))
        # videos.append(makeObj('RGD3T4M01H02'))
        # videos.append(makeObj('RDG3T4M01H02Sal1'))
        # videos.append(makeObj('RGD3T4M02H01'))
        # videos.append(makeObj('RGD3T4M02H01Sal'))
        # videos.append(makeObj('RGD3T4M02H02'))
        # videos.append(makeObj('RGD3T4M02H02Sal'))
        # videos.append(makeObj('RGD3T4M03H01'))
        # videos.append(makeObj('RGD3T4M03H01Sal'))
        # videos.append(makeObj('RGD3T4M03H02'))
        # videos.append(makeObj('RGD3T4M03H02Sal'))
        videos.append(makeObj('RGD3T4M06H01_2'))
        videos.append(makeObj('RGD3T4M06H01Sal_2'))
        videos.append(makeObj('RGD3T4M06H02_2'))
        videos.append(makeObj('RGD3T4M06H02Sal_2'))
        videos.append(makeObj('RGD3T4M07H01_2'))
        videos.append(makeObj('RGD3T4M07H01Sal_2'))
        videos.append(makeObj('RGD3T4M07H02_2'))
        videos.append(makeObj('RGD3T4M07H02Sal_2'))
    else:
        # ================= GD4 ==================
        # Order: name, mean_uterus_size, th_bot, th_top, only_bot):
        videos.append(makeObj('RGD4T4M01H01'))
        videos.append(makeObj('RGD4T4M01H01Sal2'))
        videos.append(makeObj('RGD4T4M01H02'))
        videos.append(makeObj('RGD4T4M02H01MdSal2'))
        videos.append(makeObj('RGD4T4M02H02MdSal2'))
        videos.append(makeObj('RGD4T4M03H01Md'))
        videos.append(makeObj('RGD4T4M03MdH01Sal2'))
        videos.append(makeObj('RGD4T4M01H02Sal2'))
        videos.append(makeObj('RGD4T4M02H02Md'))
        videos.append(makeObj('RGD4T4M02MdH01'))
        videos.append(makeObj('RGD4T4M03H02Md'))
        videos.append(makeObj('RGD4T4M03MdH02Sal2'))
        videos.append(makeObj('RGD4T4M06H01_2'))
        videos.append(makeObj('RGD4T4M06H01Sal_2'))
        videos.append(makeObj('RGD4T4M06H02_2'))
        videos.append(makeObj('RGD4T4M06SalH02_2'))
        videos.append(makeObj('RGD4T4M07H01_2'))
        videos.append(makeObj('RGD4T4M07H02_2'))
        videos.append(makeObj('RGD4T4M07SalH01_2'))
        videos.append(makeObj('RGD4T4M07SalH02_2'))
        videos.append(makeObj('RGD4T4M08H01_2'))
        videos.append(makeObj('RGD4T4M08H02_2'))
        videos.append(makeObj('RGD4T4M08H02Sal_2'))
        videos.append(makeObj('RGD4T4M08SalH01_2'))
        videos.append(makeObj('RGD4T4M09H01_2'))
        videos.append(makeObj('RGD4T4M09H01_3'))
        videos.append(makeObj('RGD4T4M09H02_2'))
        videos.append(makeObj('RGD4T4M09H02_3'))
        videos.append(makeObj('RGD4T4M09SalH01_2'))
        videos.append(makeObj('RGD4T4M09SalH01_3'))
        videos.append(makeObj('RGD4T4M09SalH02_2'))
        videos.append(makeObj('RGD4T4M09SalH02_3'))

    #f_types = ['Area', 'Bottom_positions', 'Mean_intensities', 'Top_positions']
    f_types = ['Area', 'Bottom_positions', 'Mean_intensities']

    for i, cur_vid in enumerate(videos):
        file_name = cur_vid['file_name']
        data_folder = F'{input_folder}/{gd_video}/Original/{file_name}'
        output_folder = F'{input_folder}/{gd_video}/FilteredImages/{file_name}/'
        saveDir(output_folder)

        for j, cur_f_type in enumerate(f_types):
            df = pd.read_csv(join(data_folder,file_name+'_'+cur_f_type+'.csv'), dtype=np.float32)
            rows,cols = df.shape

            original_values = df.values
            print(F'Working with file {file_name} filter {cur_f_type}')

            print(F'Saving original data...')
            # Save as JPG
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Original')
            title=F'{file_name}  {cur_f_type} Original'
            original_norm = (original_values - np.amin(original_values))/np.ptp(original_values)
            # plotFinalFigures(original_values, title, new_file_name+'.jpg',computeExtent(original_values))
            plotFinalFigures(original_values, title, new_file_name+'.jpg',[])

            # Save original data as HTML
            # title=F'{file_name} {cur_f_type}'
            # html_file_name = F'{new_file_name}.html'
            # plotHeatmatPlotty(original_values, rows, cols, title, html_file_name)

            k = 21
            # Save as Edges
            title=F'{file_name} {cur_f_type} horizontal edge '
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Horizontal_Edge_Original')
            edge = cv2.Sobel(original_values,cv2.CV_64F,0,1,ksize=k)
            plotFinalFigures(edge, title, new_file_name+'.jpg',computeExtent(edge))
            plotHeatmatPlotty(edge, rows, cols, title, new_file_name+'.html')
            # np.savetxt(F'{new_file_name}.csv', edge,fmt='%10.3f', delimiter=',')

            k = 31
            title=F'{file_name} {cur_f_type} vertical edge original'
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Vertical_Edge')
            edge = cv2.Sobel(original_values,cv2.CV_64F, 1, 0,ksize=k)
            edge = (edge - np.amin(edge))/np.ptp(edge)
            # plotFinalFigures(edge, title, new_file_name+'.jpg',computeExtent(edge))
            plotFinalFigures(edge, title, new_file_name+'.jpg',[])
            plotHeatmatPlotty(edge, rows, cols, title, new_file_name+'.html')
            np.savetxt(F'{new_file_name}.csv', edge,fmt='%10.3f', delimiter=',')

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
            title=F'{file_name}   Filtered'
            plotFinalFigures(clean_intensities, title, new_file_name+'.jpg',computeExtent(clean_intensities))

            # Save as CSV
            np.savetxt(F'{new_file_name}.csv', clean_intensities,fmt='%10.3f', delimiter=',')

            # Save as HTML
            title=F'{file_name} {cur_f_type}'
            html_file_name = F'{new_file_name}.html'
            plotHeatmatPlotty(clean_intensities, rows, cols, title, html_file_name)

            # Save as Edges
            title=F'{file_name} {cur_f_type} horizontal edge filtered'
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Horizontal_Edge_Filtered')
            edge = cv2.Sobel(clean_intensities,cv2.CV_64F,0,1,ksize=k)
            plotFinalFigures(edge, title, new_file_name+'.jpg',computeExtent(edge))
            plotHeatmatPlotty(edge, rows, cols, title, new_file_name+'.html')

            title=F'{file_name} {cur_f_type} vertical edge filtered'
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Vertical_Edge_Filtered')
            edge = cv2.Sobel(clean_intensities,cv2.CV_64F,1,0,ksize=k)
            plotFinalFigures(edge, title, new_file_name+'.jpg',computeExtent(edge))
            plotHeatmatPlotty(edge, rows, cols, title, new_file_name+'.html')

            # Save as CSV
            np.savetxt(F'{new_file_name}.csv', edge,fmt='%10.3f', delimiter=',')
    print("Done!!!")
