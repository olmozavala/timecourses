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

if __name__ == '__main__':
    # data_folder = '/home/olmozavala/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Output'
    # output_folder = '/home/olmozavala/Dropbox/UMIAMI/WorkUM/DianaProjects/ContractionsFromVideos/Output/Curves'
    data_folder = '../Output'
    output_folder = '../Output/FilteredImages'

    videos=[]
    videos.append(makeObj('ASalGD3M01H02ctrl2inj11AMdis3PM_3'))
    videos.append(makeObj('GD2_11AM_2'))
    videos.append(makeObj('GD3_11AM'))
    videos.append(makeObj('GD3T4control'))
    videos.append(makeObj('NP'))
    videos.append(makeObj('RGD3T4M01H01'))
    videos.append(makeObj('RDG3TEM01H01Sal1'))

    f_types = ['Bottom_positions','Mean_intensities','Top_positions','Area']

    for i,cur_vid in enumerate(videos):
        for j,cur_f_type in enumerate(f_types):
            file_name = cur_vid['file_name']

            df = pd.read_csv(join(data_folder,file_name+'_'+cur_f_type+'.csv'), dtype=np.float32)
            rows,cols = df.shape

            clean_intensities = np.zeros((rows,cols))
            print(F'Working with file {file_name}')
            print('Smoothing the curves...')
            for frame in df.index.values:
                intensities = df.loc[frame].values
                low_freq = smoothSingleCurve(intensities, 40) # Gets low frequencies
                removed_low = intensities - low_freq
                clean_intensities[frame,:]= smoothSingleCurve(removed_low, 3) # Removes high frequencies
                # clean_intensities[frame,:]= intensities
                # clean_intensities[frame,:]= removed_low
                # clean_intensities[frame,:] = smoothSingleCurve(intensities, 4) # Removes high frequencies

                # For plotting and saving single curve
                # title=F'Frame {frame} file {file_name}'
                # plt.figure(figsize=(15,5))
                # plt.plot(range(len(clean_intensities)),clean_intensities)
                # plt.title(title)
                # plt.xlabel('Seconds')
                # plt.ylabel('Intensity')
                # plt.grid()
                # output_file_name = join(output_folder,F'{file_name}_{cur_f_type}_{frame:04d}.png')
                # plt.savefig(output_file_name)
                # # plt.show()
                # plt.close()

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

