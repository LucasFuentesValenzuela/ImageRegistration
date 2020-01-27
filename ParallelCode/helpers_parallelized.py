import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import os
from tqdm import tqdm
import tifffile as tiff
import skimage
from astroalign import *
import multiprocess as mp



def cropImage(filename, nb_pixel,delta,fov_oct,Folders,offset):
    #concatenate FOVs, then crop ROI

    pixel_size=0.365623*10**-6
    size_cropping=fov_oct+2*delta
    n_pixels_map=round(size_cropping/pixel_size)
    fov_nh_x=nb_pixel['x']*pixel_size#nh = nighthawk, i.e. high mag
    fov_nh_y=nb_pixel['y']*pixel_size

    if filename.startswith('._'):
        pass
    if 'fluorescent' in filename:
        
        print(filename)
        #assuming setup == 2
        low_mag_id_1=int(filename[1:5])#along y
        low_mag_id_2=int(filename[6:10])#along x

        full_image_name=str(low_mag_id_1)+'_'+str(low_mag_id_2)
        #TODO: change this with a boolean variable "recompute"
        # if os.path.exists(Folders['Virtual']+full_image_name+'_ch1.tif'):
        #     print("skipping")
        #     pass
        
        #assuming setup == 2
        y_start=max(max(low_mag_id_1*fov_oct-delta,0)+offset['y'],0)
        x_start=max(low_mag_id_2*fov_oct-delta,0)+offset['x'] 
        partnb=1

        y_end=y_start+fov_oct+2*delta
        x_end=x_start+fov_oct+2*delta
        
        #loop through the frames
        #identify which ones are the limit ones
        nb_im_x=23
        # print("x_s, x_e: ", x_start, x_end)
        # print("y_s, y_e: ", y_start, y_end)
        for i in range(nb_im_x):
            if i*fov_nh_x<=x_start and (i+1)*fov_nh_x>=x_start:
                frame_x_start=i+1
            if i*fov_nh_x<=x_end and (i+1)*fov_nh_x>=x_end:
                frame_x_end=i+1
            if i*fov_nh_y<=y_start and (i+1)*fov_nh_y>=y_start:
                frame_y_start=i+1
            if i*fov_nh_y<=y_end and (i+1)*fov_nh_y>=y_end:
                frame_y_end=i+1

        #create blank picture
        # print("frame y end: ", frame_y_end)
        # print("frame y start", frame_y_start)
        full_image=np.zeros((nb_pixel['y']*(frame_y_end-frame_y_start+1),nb_pixel['x']*(frame_x_end-frame_x_start+1),3),dtype='uint16')

        idx_x=np.arange(frame_x_end,frame_x_start-1,-1)
        idx_y=np.arange(frame_y_end,frame_y_start-1,-1)   

        #iterate over the different frames making up the tiling
        for i in range(idx_x.shape[0]):
            for j in range(idx_y.shape[0]):
                # filename=str(idx_y[j])+'_'+str(idx_x[i])+'.tif'
                key='HighMag_part'+str(partnb)

                for fnm in os.listdir(Folders[key]):
                    if fnm.startswith("._"):
                        continue
                    if fnm.endswith(".tif"):  
                        nbrs=fnm[-11:-4]
                        nb1=int(nbrs[0:3])
                        nb2=int(nbrs[4:8]) 

                        #assuming setup == 2
                        nb2=23-nb2

                        if nb1 == idx_y[j] and nb2==idx_x[i]:
                            filename = fnm
                            pass
                
                # print("I should have the file corresponding to ", str(idx_y[j])+'_'+str(idx_x[i])+'.tif')
                # print("I am loading ",Folders[key]+ filename)
            
                if os.path.exists(Folders[key]+filename):
                    crt_img=tiff.imread(Folders[key]+filename)

                    #flip_rotate image here
                    #much easier, as it does not require you to load and save a large tiff image another time before

                    img_list=[]
                    for channel in [0,1,2]:
                        img=crt_img[channel,:,:]
                        img=cv2.flip(img,1)
                        img=img.transpose()
                        img_list.append(img)
                    crt_img=np.dstack(img_list)

                else:
                    print("file does not exist")
                    pass
                # print(crt_img.shape)
                nb_pixel['y']=crt_img.shape[0]
                nb_pixel['x']=crt_img.shape[1]
                # print(nb_pixel)
                full_image[j*nb_pixel['y']:(j+1)*nb_pixel['y'],i*nb_pixel['x']:(i+1)*nb_pixel['x'],:]=crt_img

                #determine the shift for the crop region 
                if i==0 and j==0:
                    #this condition is to properly determine what area of the region you want to crop
                    #super important to have it correctly coded, as otherwise you risk extracting the wrong frames
                    #in this case we add 10-1 because we did the same thing with the frames above
                    #to be cleaned and improved
                    if partnb==2: 
                        x_start_crt=nb_pixel['x']*(idx_x[i]-1+10-1)*pixel_size
                    else: 
                        x_start_crt=nb_pixel['x']*(idx_x[i]-1)*pixel_size
                    y_start_crt=nb_pixel['y']*(idx_y[j]-1)*pixel_size
                    x_ref=x_start_crt+nb_pixel['x']*pixel_size
                    y_ref=y_start_crt+nb_pixel['y']*pixel_size
                    x_idx=int(round((x_ref-x_end)/pixel_size))
                    y_idx=int(round((y_ref-y_end)/pixel_size))
        
        full_image=full_image[y_idx:y_idx+n_pixels_map,x_idx:x_idx+n_pixels_map,:]
        # tiff.imsave(Folders['Virtual']+full_image_name+'.tif',full_image)

        for ch in [0,1,2]:
            cv2.imwrite(Folders['Virtual']+full_image_name+'_ch'+str(ch)+'.png',full_image[:,:,ch])
            cv2.imwrite(Folders['Virtual']+full_image_name+'_ch'+str(ch)+'.tif',full_image[:,:,ch])
    else:
        pass

# REGISTRATION

def binarize_img(img,th,hM):
    # th is between 0 and 1
    # print("setting thresholds")
    #np.iinfo: just to scale the threshold related to the actual maximum value of the encoding
    thresh=cv2.threshold(img, th*np.iinfo(img.dtype).max, 255, cv2.THRESH_BINARY)[1]
    thresh=thresh.astype('uint8')
    # print("treshold: ", thresh)
    # print(th*np.iinfo(img.dtype).max)
    # print(thresh.dtype)
    if hM==True:
        n_iter=1
        n_iter_bis=5
    else:
        n_iter=2
        n_iter_bis=4
    # print("eroding and dilating")
    thresh = cv2.erode(thresh, None, iterations=n_iter)
    thresh = cv2.dilate(thresh, None, iterations=n_iter_bis)
    
    return thresh

def get_transform_withScaling(img_virtual,img_lowMag,th,scaleUpFactor=2):

    img_virtual_bin=binarize_img(img_virtual,th,True)
    img_lowMag_bin=binarize_img(img_lowMag,th,False)

    # finding the transform
    # print("finding the transform")
    transf, (pos_img_virtual, pos_img_lM) = find_transform(img_virtual_bin,img_lowMag_bin)
    # print("transform found")

    # update transform to include scaling
    T = skimage.transform.SimilarityTransform(matrix=None, scale=transf.scale*scaleUpFactor, rotation=transf.rotation, translation=transf.translation*scaleUpFactor)
    # set target size
    target = np.empty(tuple(np.round(scaleUpFactor*np.array(img_lowMag.shape)).astype(int)))

    return T, target

def viz_ref_points(img_virtual,img_lowMag,img_aligned,transf,pos_img_virtual,pos_img_lM,r, circles=True):
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

    scaleUpFactor=3.2
    if filename.startswith('._'):
        return
    if 'fluorescent' in filename:
        print(filename)
        low_mag_id_1=int(filename[1:5])#along y
        low_mag_id_2=int(filename[6:10])#along x

        filename_highmag_prefix=str(low_mag_id_1)+'_'+str(low_mag_id_2)
        filename_highmag_prefix_zfill = str(low_mag_id_1).zfill(4) + '_'+ str(low_mag_id_2).zfill(4)
        filename_virtual=Folders['Virtual']+filename_highmag_prefix+'_ch1'+str('.tif')
        filename_lowMag=Folders['lowMag']+filename
        img_virtual=cv2.imread(filename_virtual,cv2.IMREAD_UNCHANGED)
        img_lowMag=cv2.imread(filename_lowMag,0)

        try: 
            print('    trying to find a transform for ' + filename_highmag_prefix + ' ... ')
            T,target = get_transform_withScaling(img_virtual,img_lowMag,th,scaleUpFactor)
            for ch in range(3):
                filename_highMag=filename_highmag_prefix + '_ch' + str(ch) + str('.tif')
                filename_highMag_zfill=filename_highmag_prefix_zfill + '_ch' + str(ch) + str('.tif')
                print('    applying the transform to ' + filename_highMag)
                image_toBeAligned=cv2.imread(Folders['Virtual']+filename_highMag,cv2.IMREAD_UNCHANGED)
                cv2.imwrite(Folders['Aligned']+filename_highMag_zfill,apply_transform(T,image_toBeAligned,target)[0].astype('uint16'))

        except MaxIterError: 
            # print('ERROR: Could not find Registration')
            # missed_FOVs.append(filename)
            print('Error - max iteration reached for ' + filename_highmag_prefix)
            return
        except TooFewPointsError: 
            # print('Too Few Points')
            # missed_FOVs.append(filename)
            print('Error - too few points for ' + filename_highmag_prefix)
            return
        except: #Try to understand what this is
            # print('Other Error')
            # missed_FOVs.append(filename)
            print('Error - other error for ' + filename_highmag_prefix)
            return
    else:
        return

from functools import partial

def run_concatenate_crop_parallelized(nb_pixel,delta,fov_oct,Folders,offset,numThreads=16):
    print('running concatenate_crop')
    filename_list = tqdm(os.listdir(Folders['lowMag']))
    pool = mp.Pool(processes=numThreads)
    pool.map(partial(cropImage,nb_pixel=nb_pixel,delta=delta,fov_oct=fov_oct,Folders=Folders,offset=offset), filename_list)

def run_registration_parallelized(Folders,th,numThreads=1):
    print('running image registration')
    filename_list = tqdm(os.listdir(Folders['lowMag']))
    pool = mp.Pool(processes=numThreads)
    pool.map(partial(alignImage,Folders=Folders,th=th), filename_list)