import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import os
import tifffile as tiff
from helpers_parallelized import *
from astroalign import *

Volume='/Volumes/Extreme SSD/16C_1part/NewCode/'
Volume2='/Volumes/Extreme SSD/16C_1part/'

Folders=dict()
Folders['HighMag_part1']=os.path.join(Volume2,'stitched/')
Folders['lowMag']=os.path.join(Volume2,'BGremoved/')
Folders['HighMag_RF']=os.path.join(Volume,'HighMag_RF/')
Folders['Virtual']=os.path.join(Volume,'HighMag_Tiled/')
Folders['Aligned']=os.path.join(Volume,'Registered/')

# high mag pixel size
pixel_size=0.365623*10**-6

# high mag split stiched image size
nb_pixel=dict()
nb_pixel['y']=3000
nb_pixel['x']=2500

# octopi FOV
fov_oct=1.6*10**-3

#These parameters have to be determined in advance, manually
#3C
offset=dict()
offset['x']=1800*pixel_size
offset['y']=200*pixel_size

#difference on each side, manual parameter
delta=5*10**-4

# run coarse alignment
run_concatenate_crop_parallelized(nb_pixel,delta,fov_oct,Folders,offset,numThreads=16)

# run registration
th=150.0/255

# filename='_0003_0010_fluorescent.png'
# cropImage(filename, nb_pixel,delta,fov_oct,Folders,offset)
# alignImage(filename,Folders,th)

# run_registration_parallelized(Folders,th)

