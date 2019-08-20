import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from Utils_io import *
from os.path import join
import numpy as np

view_results = False
colors = ['y', 'r', 'c', 'b', 'g', 'w', 'k', 'y', 'r', 'c', 'b', 'g', 'w', 'k']

def dispImages():
    if view_results:
        plt.show()
    else:
        plt.close()

def plotMultipleImages(imgs, titles=[], output_folder='', file_name='', merge=False):
    for img_idx, c_img in enumerate(imgs):
        plt.imshow(c_img)

        if len(titles) > img_idx:
            plt.title(F'{titles[img_idx]}')

        if output_folder!='':
            plt.savefig(join(output_folder,file_name), bbox_inches='tight')

        if not(merge):
            dispImages()


    if merge:
        dispImages()

def plotImageAndScatter(img, sc_data=[], title='', savefig=False, output_folder='', file_name=''):
    cols = img.shape[1]
    plt.imshow(img)
    for c_sc in sc_data:
        plt.plot(range(cols), c_sc, 'r')

    if title != '':
        plt.title(title)

    if savefig:
        saveDir(output_folder)
        plt.savefig(join(output_folder,file_name), bbox_inches='tight')

    dispImages()

def plotImageAndMask(img, mask, title='', savefig=False, output_folder='', file_name=''):
    plt.imshow(img)
    plt.contour(mask, colors=colors[0], linewidths=.3)
    if title!='':
        plt.title()

    if savefig:
        saveDir(output_folder)
        plt.savefig(join(output_folder,file_name), bbox_inches='tight')

    dispImages()

def plotFinalFigures(data, title, out_file_name, extent=[]):
    if len(extent)>0:
        plt.imshow(data, interpolation='none', extent=extent, aspect='auto')
    else:
        plt.imshow(data, interpolation='none')
    plt.title(title)
    plt.colorbar()
    plt.gca().invert_yaxis()
    # plt.xlabel('Milimeters')
    # plt.ylabel('Seconds')
    plt.savefig(out_file_name, bbox_inches='tight')
    dispImages()

def plotHeatmatPlotty(z,rows, cols, title, filename):
    data= go.Heatmap(z=z,
                     x=np.arange(cols),
                     y=np.arange(rows))

    numticks = 10
    fps = 2
    pix_per_mil = 56
    font_size = 22

    layout= go.Layout(
                title=title,
                # xaxis=dict(
                #     title='Milimeters',
                #     ticktext=[F'{x/pix_per_mil:.0f}' for x in np.linspace(0,cols,numticks)],
                #     tickvals=np.linspace(0,cols,numticks),
                #     titlefont=dict(size=font_size)
                # ),
                # yaxis=dict(
                    # autorange='reversed',
                    # title='Seconds',
                    # ticktext=[F'{x/fps:.0f}' for x in np.linspace(0,rows,numticks)],
                    # tickvals=np.linspace(0,rows,numticks),
                    # titlefont=dict(size=font_size)
                # )
    )


    fig = go.Figure(data=[data],layout=layout)
    plotly.offline.plot(fig, filename=filename, auto_open=False)

