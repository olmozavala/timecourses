Install
========

2 frames por segundo

Conda packages
-------
```
conda install -c conda-forge opencv
conda install -c conda-forge/label/broken opencv
conda install -c plotly plotly
```


Mask selection
--------------

The mask selection uses the following steps:

Smooth the image to compute Sobel with Gaussian blur of size 11x11 for
each frame (smoothing rows and columns) and for each column 
(smoothing rows and frames)

Compute Sobel edge detection on X and Y with a window size of 5x5. 
Compute final edge detector with an emphasis on the horizontal detection
with final_sobel = 2*sobel_x + sobel_y

Select the top positions and bottom positions by selecting,
for each column, the row with the highest and lowest edge detection
value. 

The final positions get further smoothed by approximating
a cubic function. For the top positions the 
 total number of knot points used is 20 (because it is harder
 to detect the proper boundary of the uterus), for the bottom
 positions the total number of knots is 40.  
 
 Mean intensities and area is computed from those two. 
