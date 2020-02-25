import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import os
from tqdm import tqdm
import tifffile as tiff
import skimage
from astroalign import *
import multiprocessing as mp



def cropImage(filename, nb_pixel, delta, fov_oct, Folders, offset, recompute_all=True, setup=1):
    """
    Computes a coarse alignment around the lowMag image. 

    Inputs
    ------
    filename: str
        name of the Low Mag image used as a reference
    nb_pixel: dict
        dictionnary containing the size (in pixels) of the HighMag images
        keys are ('x', 'y')
    delta: float
        size of the margins for the tiling, to account for offset error
    fov_oct: float 
        size of the low mag images (in meters)
    Folders: dict  
        contains the in and out paths for the different folders
    offset: dict
        contains the offset (in pixels) along both directions
        keys are ('x', 'y')
    
    Outputs
    -------
    Does not return anything
    Writes tiled images in specified folders
    """

    pixel_size=0.365623*10**-6
    size_cropping=fov_oct+2*delta
    n_pixels_map=round(size_cropping/pixel_size)
    fov_nh_x=nb_pixel['x']*pixel_size#nh = nighthawk, i.e. high mag
    fov_nh_y=nb_pixel['y']*pixel_size

    if filename.startswith('._'):
        return
    if 'fluorescent' in filename:
        
        print(filename)
        parse_fn=filename.split('_')
        
        #setup: there was setup here but don't think needed now
        low_mag_id_1=int(parse_fn[1])#along y
        low_mag_id_2=int(parse_fn[2])#along x
        full_image_name=str(low_mag_id_1)+'_'+str(low_mag_id_2)

        #Skip the cropping if there is already a version of it and if we do not want to 
        #recompute everything from scratch
        if os.path.exists(Folders['Virtual']+full_image_name+'_ch1.tif') and not recompute_all:
            print("Skipping this frame, as a version already exists.")
            return

        #Location of the start and end of the frame 
        y_start=max(max(low_mag_id_1*fov_oct-delta,0)+offset['y'],0)
        x_start=max(max(low_mag_id_2*fov_oct-delta,0)+offset['x'],0)
        y_end=y_start+fov_oct+2*delta
        x_end=x_start+fov_oct+2*delta 

        # print("x start: ", x_start)
        # print("y start: ", y_start)
        #loop through the frames
        #identify which ones are the limit ones
        nb_im_x=23 #maximum number of frames
        nb_im_y=19
        # for i in range(nb_im):
        #     if i*fov_nh_x<=x_start and (i+1)*fov_nh_x>=x_start:
        #         frame_x_start=i+1
        #     if i*fov_nh_x<=x_end and (i+1)*fov_nh_x>=x_end:
        #         frame_x_end=i+1
        #     if i*fov_nh_y<=y_start and (i+1)*fov_nh_y>=y_start:
        #         frame_y_start=i+1
        #     if i*fov_nh_y<=y_end and (i+1)*fov_nh_y>=y_end:
        #         frame_y_end=i+1

        frame_x_start=int(np.floor(x_start/fov_nh_x))+1
        frame_x_end=np.minimum(int(np.ceil(x_end/fov_nh_x))+1, nb_im_x)
        frame_y_start=int(np.floor(y_start/fov_nh_y))+1
        frame_y_end=np.minimum(int(np.ceil(y_end/fov_nh_y))+1, nb_im_y)
        # print(frame_x_start, frame_x_end)
        # print(frame_y_start, frame_y_end)
        #create empty image to be filled in later
        full_image=np.zeros((nb_pixel['y']*(frame_y_end-frame_y_start+1),nb_pixel['x']*(frame_x_end-frame_x_start+1),3),dtype='uint16')

        #Determine the IDs of the high mag frames to load for the tiling
        idx_x=np.arange(frame_x_end,frame_x_start-1,-1)
        idx_y=np.arange(frame_y_end,frame_y_start-1,-1)   

        #iterate over the different frames making up the tiling
        for i in range(idx_x.shape[0]):
            for j in range(idx_y.shape[0]):
                # filename=str(idx_y[j])+'_'+str(idx_x[i])+'.tif'
                key='HighMag'

                for fnm in os.listdir(Folders[key]):
                    if fnm.startswith("._"):
                        continue
                    if fnm.endswith(".tif"):  
                        #TODO: make this clearer
                        nbrs=fnm[-11:-4]
                        nb1=int(nbrs[0:3])
                        nb2=int(nbrs[4:8]) 

                        
                        if setup==1:#like 3C
                            pass #the system is built for that kind of numbering
                        elif setup==2: #if the numbering is different
                            nb2=23-nb2

                        if nb1 == idx_y[j] and nb2==idx_x[i]:
                            filename = fnm
                            pass
                
                if os.path.exists(Folders[key]+filename):
                    crt_img=tiff.imread(Folders[key]+filename)

                    #flip_rotate image here
                    #much easier, as it does not require you to load and save a large tiff image another time before

                    img_list=[]
                    for channel in [0,1,2]:
                        img=crt_img[channel,:,:]
                        if setup==1:
                            img=img.transpose()
                            img=cv2.flip(img,0)
                            img=cv2.flip(img,1)
                        elif setup==2:
                            img=cv2.flip(img,1)
                            img=img.transpose()
                        img_list.append(img)
                    crt_img=np.dstack(img_list)

                else:
                    print("file does not exist")
                    return


                nb_pixel['y']=crt_img.shape[0]
                nb_pixel['x']=crt_img.shape[1]
                full_image[j*nb_pixel['y']:(j+1)*nb_pixel['y'],i*nb_pixel['x']:(i+1)*nb_pixel['x'],:]=crt_img

                #determine the shift for the crop region 
                if i==0 and j==0:
                    #this condition is to properly determine what area of the region you want to crop
                    #super important to have it correctly coded, as otherwise you risk extracting the wrong frames
                    #in this case we add 10-1 because we did the same thing with the frames above
                    #to be cleaned and improved
                    x_start_crt=nb_pixel['x']*(idx_x[i]-1)*pixel_size
                    y_start_crt=nb_pixel['y']*(idx_y[j]-1)*pixel_size
                    x_ref=x_start_crt+nb_pixel['x']*pixel_size
                    y_ref=y_start_crt+nb_pixel['y']*pixel_size
                    x_idx=int(round((x_ref-x_end)/pixel_size))
                    y_idx=int(round((y_ref-y_end)/pixel_size))

                    # print("idx: ", x_idx, y_idx)
                    #sometimes you just don't have the frame (x_idx is negative)
                    x_idx=np.maximum(x_idx,0)
                    y_idx=np.maximum(y_idx,0)
        
        # print(full_image.shape)
        # print(y_idx+n_pixels_map)
        # print(x_idx+n_pixels_map)
        full_image=full_image[y_idx:y_idx+n_pixels_map,x_idx:x_idx+n_pixels_map,:]
        # tiff.imsave(Folders['Virtual']+full_image_name+'.tif',full_image)

        for ch in [0,1,2]:
            # cv2.imwrite(Folders['Virtual']+full_image_name+'_ch'+str(ch)+'.png',full_image[:,:,ch])
            cv2.imwrite(Folders['Virtual']+full_image_name+'_ch'+str(ch)+'.tif',full_image[:,:,ch])
    else:
        return

# REGISTRATION

def binarize_img(img,th,hM):
    """
    Computes a binarized version of the image in order to find the transform. 

    Inputs
    ------
    img: np.array
        The image to be binarized
    th: float
        The threshold used to determine 0-1 in the image
    hM: bool
        Whether it is a highmag image

    Outputs
    -------
    thresh: np.array
        the thresholded image (after erosion and dilation)

    """
    # th is between 0 and 1
    #np.iinfo: just to scale the threshold related to the actual maximum value of the encoding
    thresh=cv2.threshold(img, th*np.iinfo(img.dtype).max, 255, cv2.THRESH_BINARY)[1]

    thresh=thresh.astype('uint8')

    if hM==True:
        n_iter=1
        n_iter_bis=5
    else:
        n_iter=2
        n_iter_bis=4
    thresh = cv2.erode(thresh, None, iterations=n_iter)
    thresh = cv2.dilate(thresh, None, iterations=n_iter_bis)
    
    return thresh

def get_transform_withScaling(img_virtual,img_lowMag,th,scaleUpFactor=2):
    """
    Computes the transform between the low mag and the high mag image. 

    Inputs
    ------
    img_virtual: np.array
        The result of the tiling of multiple highmag frames.
    img_lowMag: np.array
        The original lowMag image
    th: float
        The threshold used for binarizing the image
    scaleUpFactor: float
        TBD


    Outputs
    -------
    T: SimilarityTransform
        A similarity transform between the low and high mag images (binarized)
    target: np.array
        An empty array, whose size is the size of the desired registered image. 

    """
    img_virtual_bin=binarize_img(img_virtual,th,True)
    img_lowMag_bin=binarize_img(img_lowMag,th,False)
    # finding the transform
    transf, (pos_img_virtual, pos_img_lM) = find_transform(img_virtual_bin,img_lowMag_bin)

    # update transform to include scaling
    T = skimage.transform.SimilarityTransform(matrix=None, scale=transf.scale*scaleUpFactor, rotation=transf.rotation, translation=transf.translation*scaleUpFactor)

    # set target size
    target = np.empty(tuple(np.round(scaleUpFactor*np.array(img_lowMag.shape)).astype(int)))

    return T, target

def viz_ref_points(img_virtual,img_lowMag,img_aligned,transf,pos_img_virtual,pos_img_lM,r, circles=True):
    """
    Viz routine in order to compare points from the original and the target images in the registration. 
    Very useful for debugging. 
    """
    _, axes = plt.subplots(2, 2, figsize=(10, 10))

    colors = ['r', 'g', 'b', 'y', 'cyan', 'w', 'm']

    axes[0, 0].imshow(img_virtual, cmap='gray', interpolation='none', vmin=0, vmax=255)
    axes[0, 0].axis('off')
    axes[0, 0].set_title("Source Image")
    if circles ==True:
        for (xp, yp), c in zip(pos_img_virtual[:len(colors)], colors):
            circ = plt.Circle((xp, yp), r, fill=False, edgecolor=c, linewidth=2)
            axes[0, 0].add_patch(circ)

    axes[0, 1].imshow(img_lowMag, cmap='gray', interpolation='none', vmin=0, vmax=255)
    axes[0, 1].axis('off')
    axes[0, 1].set_title("Target Image")
    if circles==True:
        for (xp, yp), c in zip(pos_img_lM[:len(colors)], colors):
            circ = plt.Circle((xp, yp), r * transf.scale, fill=False, edgecolor=c, linewidth=2)
            axes[0, 1].add_patch(circ)

    axes[1, 1].imshow(img_aligned, cmap='gray', interpolation='none', vmin=0, vmax=255)
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Source Image aligned with Target")
    if circles ==True:
        for (xp, yp), c in zip(pos_img_lM[:len(colors)], colors):
            circ = plt.Circle((xp, yp), r * transf.scale, fill=False, edgecolor=c, linewidth=2)
            axes[1, 1].add_patch(circ)

    axes[1, 0].axis('off')

    plt.tight_layout()
    plt.show()

def alignImage(filename,Folders,th):
    """
    Compute the registration. 

    Inputs
    ------
    filename: str
        Name of the low mag image. Warning: the string format is important for the code to work. 
    Folders: dict
        Dictionnary containing the different paths for the files (input and output)
    th: float
        threshold for the binarization of the image. 

    Outputs
    -------
    Does not output anything.
    Writes registered images to a specified folder. 
    """

    #value: TBD
    scaleUpFactor=3.2

    if filename.startswith('._'):
        return
    if 'fluorescent' in filename:

        print(filename)
        parse_fn=filename.split('_')
        low_mag_id_1=int(parse_fn[1])#along y
        low_mag_id_2=int(parse_fn[2])#along x

        filename_highmag_prefix=str(low_mag_id_1)+'_'+str(low_mag_id_2)
        filename_highmag_prefix_zfill = str(low_mag_id_1).zfill(4) + '_'+ str(low_mag_id_2).zfill(4)
        filename_virtual=Folders['Virtual']+filename_highmag_prefix+'_ch1'+str('.tif')
        filename_lowMag=Folders['lowMag']+filename

        #read images
        img_virtual=cv2.imread(filename_virtual,cv2.IMREAD_UNCHANGED)
        img_lowMag=cv2.imread(filename_lowMag,0)

        #Attempt to find a transform
        try: 
            # print('    trying to find a transform for ' + filename_highmag_prefix + ' ... ')
            T,target = get_transform_withScaling(img_virtual,img_lowMag,th,scaleUpFactor)
            for ch in range(3):
                filename_highMag=filename_highmag_prefix + '_ch' + str(ch) + str('.tif')
                filename_highMag_zfill=filename_highmag_prefix_zfill + '_ch' + str(ch) + str('.tif')
                # print('    applying the transform to ' + filename_highMag)
                image_toBeAligned=cv2.imread(Folders['Virtual']+filename_highMag,cv2.IMREAD_UNCHANGED)
                cv2.imwrite(Folders['Aligned']+filename_highMag_zfill,apply_transform(T,image_toBeAligned,target)[0].astype('uint16'))

        #Possible exceptions
        #   Too many iterations needed
        #   Too few points in the image to actually find a correspondance
        except MaxIterError: 
            print('Error - max iteration reached for ' + filename_highmag_prefix)
            return
        except TooFewPointsError: 
            print('Error - too few points for ' + filename_highmag_prefix)
            return
        except: #Try to understand what this is
            print('Error - other error for ' + filename_highmag_prefix)
            return
    else:
        return

from functools import partial

def run_concatenate_crop_parallelized(nb_pixel,delta,fov_oct,Folders,offset,recompute_all=True, setup=1,numThreads=16):
    print('running concatenate_crop')
    #there was tqdm before
    filename_list = os.listdir(Folders['lowMag'])
    pool = mp.Pool(processes=numThreads)
    pool.map(partial(
        cropImage,nb_pixel=nb_pixel,delta=delta,
        fov_oct=fov_oct,Folders=Folders,
        offset=offset,recompute_all=recompute_all,
        setup = setup
        ), filename_list)

def run_registration_parallelized(Folders,th,numThreads=16):
    print('running image registration')
    #there was tqdm before
    filename_list = os.listdir(Folders['lowMag'])
    pool = mp.Pool(processes=numThreads)
    pool.map(partial(alignImage,Folders=Folders,th=th), filename_list)