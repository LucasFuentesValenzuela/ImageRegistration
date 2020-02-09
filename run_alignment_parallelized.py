import re
import os
from helpers_parallelized import run_concatenate_crop_parallelized, run_registration_parallelized

def yes_no(question):
    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])
     
    while True:
        choice = input(question).lower()
        if choice in yes:
           return True
        elif choice in no:
           return False
        else:
           print("Please respond with 'yes' or 'no'")

def main():

    #Define paths
    Volume='/Volumes/Extreme SSD/16C_1part/NewCode/'
    Volume2='/Volumes/Extreme SSD/16C_1part/'

    Folders=dict()
    Folders['HighMag']=os.path.join(Volume2,'stitched/')
    Folders['lowMag']=os.path.join(Volume2,'BGremoved/')
    Folders['Virtual']=os.path.join(Volume,'HighMag_Tiled/')
    Folders['Aligned']=os.path.join(Volume,'Registered/')


    ##################################################
    # Define parameters

    # high mag pixel size
    pixel_size=0.365623*10**-6

    # high mag split stiched image size
    nb_pixel=dict()
    nb_pixel['y']=3000
    nb_pixel['x']=2500

    # octopi FOV
    fov_oct=1.6*10**-3

    #These parameters have to be determined in advance, manually
    offset=dict()
    offset['x']=1800*pixel_size
    offset['y']=200*pixel_size

    th=150.0/255
    delta=5*10**-4


    ################################################
    # Get input

    cropping=yes_no("Do you want to run the cropping? y/n \n")
    if cropping ==True:
        print("     Do you want to run the cropping on all images? y/n \n")
        print("         y: all images, even those that already exist in the target folder\n")
        print("         n: only the images that do not already exist in the target folder\n")
        recompute_all=yes_no("")
    registration=yes_no("Do you want to run the registration? y/n\n")

    #################################################
    # Run

    # run coarse alignment
    if cropping:
        print("---------- Running concatenation and cropping")
        run_concatenate_crop_parallelized(
            nb_pixel,delta,fov_oct,Folders,offset,
            recompute_all=recompute_all,numThreads=16
            )

    #run fine alignment
    if registration:
        print("---------- Running registration")
        run_registration_parallelized(Folders,th)

if __name__ == "__main__":
    main()

