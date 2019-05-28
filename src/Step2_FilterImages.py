from os.path import join

import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from FiltersAndUtils import *

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
    # This is the second file that needs to be executed. It substract the low frequencies and removes high frecuencies.

    # data_folder = '/home/olmozavala/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Output'
    # output_folder = '/home/olmozavala/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Output/Curves'
    data_folder = '../Output/Original'
    output_folder = '../Output/FilteredImages'

    # video_names = ['RDG3T4M01H01Sal1','RDG3T4M01H02Sal1','RGD3T4M01H01','RGD3T4M01H02','RGD3T4M02H01',
    #                'RGD3T4M02H01Sal','RGD3T4M02H02','RGD3T4M02H02Sal','RGD3T4M03H01','RGD3T4M03H01Sal',
    #                'RGD3T4M03H02','RGD3T4M03H02Sal','RGD4T4M01H01','RGD4T4M01H01Sal2']

    video_names = [ 'RGD4T4M01H01.avi','RGD4T4M02H01MdSal2.avi','RGD4T4M03H01Md.avi','RGD4T4M01H01Sal2.avi',
                    'RGD4T4M02H02Md.avi','RGD4T4M03H02Md.avi','RGD4T4M01H02.avi','RGD4T4M02H02MdSal2.avi',
                    'RGD4T4M03MdH01Sal2.avi','RGD4T4M01H02Sal2.avi','RGD4T4M02MdH01.avi','RGD4T4M03MdH02Sal2.avi']

    videos=[ makeObj(x) for x in video_names ]

    # f_types = ['Bottom_positions','Mean_intensities','Top_positions','Area']
    f_types = ['Area']

    for i,cur_vid in enumerate(videos):
        for j,cur_f_type in enumerate(f_types):
            file_name = cur_vid['file_name']

            df = pd.read_csv(join(data_folder,file_name+'_'+cur_f_type+'.csv'), dtype=np.float32)
            rows,cols = df.shape

            clean_intensities = np.zeros((rows,cols))
            print(F'Working with file {file_name} filter {cur_f_type}')
            print('Smoothing the curves...')

            # bandPassFilter(df) # If we want to properly do a bandpass filter

            intensities = cv2.GaussianBlur(intensities, (5, 5), 0)
            for frame in df.index.values:
                intensities = df.loc[frame].values
                low_freq = smoothSingleCurve(intensities, 40) # Gets low frequencies
                # removed_low = intensities - low_freq
                # clean_intensities[frame,:]= smoothSingleCurve(removed_low, 3) # Removes high frequencies
                # clean_intensities[frame,:]= intensities
                # clean_intensities[frame,:]= removed_low
                # clean_intensities[frame,:] = smoothSingleCurve(intensities, 4) # Removes high frequencies

                # For plotting and saving single curve
                title=F'Frame {frame} file {file_name}'
                plt.figure(figsize=(15,5))
                plt.plot(range(len(clean_intensities)),clean_intensities)
                plt.title(title)
                plt.xlabel('Seconds')
                plt.ylabel('Intensity')
                plt.grid()
                output_file_name = join(output_folder,F'{file_name}_{cur_f_type}_{frame:04d}.png')
                plt.savefig(output_file_name)
                plt.show()
                plt.close()

            # Using plotly
            new_file_name = join(output_folder,F'{file_name}_{cur_f_type}_Filtered')
            plt.matshow(clean_intensities)
            plt.savefig(F'{new_file_name}.jpg', bbox_inches='tight')
            plt.close()
            np.savetxt(F'{new_file_name}.csv', clean_intensities,fmt='%10.3f', delimiter=',')

            data= go.Heatmap(z=clean_intensities,
                               x=np.arange(cols),
                               y=np.arange(rows))
            layout= go.Layout(
                        title=F'{file_name}_{cur_f_type}'
                        )

            fig = go.Figure(data=[data],layout=layout)
            plotly.offline.plot(fig, filename=F'{new_file_name}.html', auto_open=False)

    print("Done!!!")
