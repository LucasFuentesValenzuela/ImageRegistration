import re
import os
from helpers_parallelized import alignImage 
import argparse

#Define paths
Volume='/Volumes/Extreme SSD/w0206_2/'
Volume2='/Volumes/GoogleDrive/My Drive/Data/w0206_2_High_Mag_stitched/'

Folders=dict()
Folders['HighMag']=os.path.join(Volume2)
Folders['lowMag']=os.path.join(Volume,'BGremoved/')
Folders['Virtual']=os.path.join(Volume,'HighMag_Tiled/')
Folders['Aligned']=os.path.join(Volume,'Registered/')

alignImage('_0009_0005_fluorescent.png',Folders=Folders,th=150.0/255)