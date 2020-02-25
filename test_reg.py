import re
import os
from helpers_parallelized import alignImage, cropImage 
import argparse

#Define paths
Volume='/Volumes/Extreme SSD/w0206_1'

Folders=dict()
Folders['HighMag']=os.path.join(Volume, 'w0206_1_HighMag/')
Folders['lowMag']=os.path.join(Volume,'w0206_1_LowMag','BGremoved/')
Folders['Virtual']=os.path.join(Volume,'HighMag_Tiled/')
Folders['Aligned']=os.path.join(Volume,'Registered/')

# high mag pixel size
pixel_size=0.365623*10**-6

nb_pixel=dict()
nb_pixel['y']=3000
nb_pixel['x']=2500
offset=dict()
offset['x']=3200*pixel_size
offset['y']=-1500*pixel_size
fov_oct=1.6*10**-3
delta=5*10**-4

cropImage('_0002_0002_fluorescent.png', nb_pixel, delta, fov_oct, Folders, offset, recompute_all=True, setup=1)
# alignImage('_0009_0005_fluorescent.png',Folders=Folders,th=150.0/255)