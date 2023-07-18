import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from Utils_io import *
from os.path import join
import numpy as np

colors = ['y', 'r', 'c', 'b', 'g', 'w', 'k', 'y', 'r', 'c', 'b', 'g', 'w', 'k']

def dispImages(view_results):
    if view_results:
        plt.show()
    else:
        plt.close()

def plotMultipleImages(imgs, titles=[], output_folder='', file_name='', 
                       view_results=True, extent=[],
                       units=[], cbar_label = [], horizontal=True, flip=False):

    font_size = 20
    if horizontal:
        fig, axs = plt.subplots(1, len(imgs), figsize=(8*len(imgs), 8))
    else:
        fig, axs = plt.subplots(len(imgs), 1, figsize=(8, 8*len(imgs)))

    for img_idx, c_img in enumerate(imgs):
        if len(imgs) == 1:
            c_ax = axs
        else:
            c_ax = axs[img_idx]

        if len(extent)>0:
            im = c_ax.imshow(c_img, extent=extent)
        else:
            im = c_ax.imshow(c_img)
        
        c_ax.set_aspect('auto')

        if flip:
            c_ax.invert_yaxis()  # This line inverts the y-axis

        if len(units)>0:
            c_ax.set_xlabel(units[0], fontsize=font_size)
            c_ax.set_ylabel(units[1], fontsize=font_size)
            c_ax.tick_params(axis='both', which='major', labelsize=font_size*.8)

        if len(titles) > img_idx:
            c_ax.set_title(F'{titles[img_idx]}', fontsize=font_size*1.2)

        shrink = 0.9
        if len(cbar_label)>0:
            fig.colorbar(im, ax=c_ax, shrink=shrink, label=cbar_label[img_idx])

        # Draw a vertical line at the middle column of the image
        # c_ax.plot([6.4, 6.4], [3, 0], 'r')


    if output_folder!='':
        plt.savefig(join(output_folder,file_name), bbox_inches='tight')

    dispImages(view_results)

def plotImageAndScatter(img, sc_data=[], title='', savefig=False, 
                        extent=[], units=[], 
                        output_folder='', file_name='', view_results=True):
    cols = img.shape[1]
    rows = img.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    if len(extent)>0:
        # plt.imshow(img, extent=extent, aspect='auto')
        plt.imshow(img, extent=extent)
    else:
        plt.imshow(img)
    if len(units)>0:
        ax.set_xlabel(units[0])
        ax.set_ylabel(units[1])

    if len(extent)>0:
        for c_sc in sc_data:
            ax.plot(np.linspace(extent[0], extent[1], cols), extent[2]*c_sc/rows, 'r')
    else:
        for c_sc in sc_data:
            ax.plot(range(cols), c_sc, 'r')

    if title != '':
        ax.set_title(title)

    if savefig:
        saveDir(output_folder)
        plt.savefig(join(output_folder,file_name), bbox_inches='tight')

    dispImages(view_results)

def plotImageAndMask(img, mask, title='', savefig=False, output_folder='', file_name='', view_results=True):
    plt.imshow(img)
    plt.contour(mask, colors=colors[0], linewidths=.3)
    if title!='':
        plt.title()

    if savefig:
        saveDir(output_folder)
        plt.savefig(join(output_folder,file_name), bbox_inches='tight')

    dispImages(view_results)

def plotFinalFigures(data, title, out_file_name, extent=[], view_results=True):
    if len(extent)>0:
        plt.imshow(data, interpolation='none', extent=extent, aspect='auto')
    else:
        plt.imshow(data, interpolation='none')
    plt.title(title, fontsize=20)
    plt.colorbar()
    plt.gca().invert_yaxis()
    # plt.xlabel('Milimeters')
    # plt.ylabel('Seconds')
    plt.savefig(out_file_name, bbox_inches='tight')
    dispImages(view_results)

def plotHeatmatPlotty(z,rows, cols, title, filename, surface=False, zmin=None, zmax=None):
    print(f"File {filename} min value {np.min(z)} max value {np.max(z)}")
    if surface:
        data= go.Surface(z=z,
                        x=np.arange(cols),
                        y=np.arange(rows))

        # X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        # data = go.Scatter3d( z=z.ravel(),
        #                 x=X.ravel(),
        #                 y=Y.ravel(),
        #                 mode='markers')
    else:
        data= go.Heatmap(z=z,
                        x=np.arange(cols),
                        y=np.arange(rows))

        if zmin is not None and zmax is not None:
            data.update(zmin=zmin, zmax=zmax)

    # Modify the min max range of the heatmap in order to see the data better

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