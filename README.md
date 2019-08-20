Install
========


Conda packages
-------
```
conda install -c conda-forge opencv
conda install -c conda-forge/label/broken opencv
conda install -c plotly plotly
```

Summary
-------

Information 
_______________
* Videos where taken at two frames per second.
* 56 pixels milimeters each direction corresponds to 1 mm.


Mask selection
_______________

The first step in the preprocessing of the videos pertains to the automatic
selection of the upper and lower boundaries of the uterus at each frame.
This is not an easy task for areas where the uterus horn is so thin
that produces a small gradient on the intensities of the image,
 making it troublesome to identify the precise location where the 
 uterus horn begins. Additionally, blood may stain the petri dish,
 adding new intensity gradients on the videos that can also mislead the 
 staring location of the horn. 
 
To obtain the boundaries of the horns, we first smooth the videos 
in the horizontal and time dimensions to remove any high-frequency 
artifacts that can affect the automatic detection. The
smoothing is made with a Gaussian filter of size 3 in time, and 
with a Gaussian filter of size 5 in space (horizontal direction only).
These parameters were selected manually and showed the best results
by subjectively comparing the original frames of the videos
 with the smoothed ones. Assuring that the artifacts were removed
but the fluctuation in the intensities caused by movement was still
 preserved. It is important to mention that the smoothing of the
 intensities in the videos it is only used to properly select the
 boundaries of the uterus horn. 

Later, the Sobel filter is used  to obtain horizontal and vertical
edges on each frame. The size of the Sobel filter used is 5x5 and,
 in order to weight more horizontal edges, the final edge computed
 is given by edge= 2*sobel_horizontal + sobel_vertical

<!-- ![](EdgesExample.jpg "Example of the solbel filter") -->

Select the top positions and bottom positions by selecting,
for each column, the row with the highest and lowest edge detection
value. 

The final positions get further smoothed by approximating
a cubic function. For the top positions the 
 total number of knot points used is 20 (because it is harder
 to detect the proper boundary of the uterus), for the bottom
 positions the total number of knots is 40.  
 
 Mean intensities and area is computed from those two. 
