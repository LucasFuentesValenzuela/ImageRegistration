import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import os

import tifffile as tiff

from astroalign_test import *

def flip_rotate_images(Folders,partnb):

    assert partnb in [1,2]
    key='HighMag_part'+str(partnb)
    print("Performing RF in following directory:", Folders[key])

    for filename in os.listdir(Folders[key]):
        
        if filename.endswith(".tif"): 
            print(filename)
            #read only the first channel, which corresponds to DAPI
            img=tiff.imread(Folders[key]+filename)
            img=img[1,:,:]
            img=img.transpose()
            img=cv2.flip(img,0)
            img=cv2.flip(img,1)
            nbrs=filename[-11:-4]
            nb1=int(nbrs[0:3])
            nb2=int(nbrs[4:8])
            new_name=str(nb1)+'_'+str(nb2)+'.tif'
            cv2.imwrite(Folders['HighMag_RF']+'part_'+str(partnb)+'/'+new_name,img)
        else:
            continue



def concatenate_crop_ROI(nb_pixel,delta,fov_oct,Folders,offset):
    #concatenate FOVs, then crop ROI

    pixel_size=0.365623*10**-6
    size_cropping=fov_oct+2*delta
    n_pixels_map=round(size_cropping/pixel_size)
    fov_nh_x=nb_pixel['x']*pixel_size#nh = nighthawk, i.e. high mag
    fov_nh_y=nb_pixel['y']*pixel_size

    for filename in os.listdir(Folders['lowMag']):
        if 'fluorescent' in filename:
            
            print(filename)
            low_mag_id_1=int(filename[1:3])#along y
            low_mag_id_2=int(filename[4:6])#along x

            #here, we apply the offset depending on which part of the scan we are using
            #it is a temporary fix until we can to handle the overlap properly
            #it needs a bit more thinking
            if low_mag_id_2 < 5:
                y_start=max(low_mag_id_1*fov_oct-delta,0)+offset['y']
                x_start=max(low_mag_id_2*fov_oct-delta,0)+offset['x']
                partnb=1
            elif low_mag_id_2 >6 : 
                y_start=max(low_mag_id_1*fov_oct-delta,0)+offset['y']#-offset['y_part2']
                x_start=max(low_mag_id_2*fov_oct-delta,0)+offset['x']#-offset['x_part2']
                partnb=2
            else:
                continue #currently ignoring some files


            y_end=y_start+fov_oct+2*delta
            x_end=x_start+fov_oct+2*delta
            
            #loop through the frames
            #identify which ones are the limit ones
            nb_im_x=21

            for i in range(nb_im_x):
                if i*fov_nh_x<=x_start and (i+1)*fov_nh_x>=x_start:
                    frame_x_start=i+1
                if i*fov_nh_x<=x_end and (i+1)*fov_nh_x>=x_end:
                    frame_x_end=i+1
                if i*fov_nh_y<=y_start and (i+1)*fov_nh_y>=y_start:
                    frame_y_start=i+1
                if i*fov_nh_y<=y_end and (i+1)*fov_nh_y>=y_end:
                    frame_y_end=i+1
            
            # print("x_frames: ", frame_x_start, frame_x_end)
            #compatibility check
            if frame_x_start<=10 and frame_x_end>10: #it needs frames from both parths
                print("This frame has not been neglected, it needs FOVs from both parts!")
                continue
            elif frame_x_start>10 and partnb==2:  #we are just changing the name for proper loading, surely to be changed afterwards when dealing with overlap
                print("x_frames: ", frame_x_start, frame_x_end)
                frame_x_start-=10
                frame_x_end-=10
                print("x_frames: ", frame_x_start, frame_x_end)
                #assuming there is an overlap of one row between part 1 and 2
                frame_x_start+=1
                frame_x_end+=1
                print("x_frames: ", frame_x_start, frame_x_end)

            

            

            
            #create blank picture
            full_image=np.zeros((nb_pixel['y']*(frame_y_end-frame_y_start+1),nb_pixel['x']*(frame_x_end-frame_x_start+1)))
            print("theory shape: ", full_image.shape)
            idx_x=np.arange(frame_x_end,frame_x_start-1,-1)
            idx_y=np.arange(frame_y_end,frame_y_start-1,-1)

            for i in range(idx_x.shape[0]):
                for j in range(idx_y.shape[0]):
                    filename=str(idx_y[j])+'_'+str(idx_x[i])+'.tif'
                    if os.path.exists(Folders['HighMag_RF']+'part_'+str(partnb)+'/'+filename):
                        crt_img=cv2.imread(Folders['HighMag_RF']+'part_'+str(partnb)+'/'+filename,0)
                    else:
                        print("file does not exist")
                        continue
                    full_image[j*nb_pixel['y']:(j+1)*nb_pixel['y'],i*nb_pixel['x']:(i+1)*nb_pixel['x']]=crt_img

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
            
            full_image=full_image[y_idx:y_idx+n_pixels_map,x_idx:x_idx+n_pixels_map]
            # print(y_idx+n_pixels_map)
            # print(x_idx+n_pixels_map)
            
            full_image_name=str(low_mag_id_1)+'_'+str(low_mag_id_2)+str('.png')
            cv2.imwrite(Folders['Virtual']+full_image_name,full_image)
            print(full_image_name)
            print(full_image.shape)
        else:
            continue




# REGISTRATION

def binarize_img(img,th,hM):
    thresh=cv2.threshold(img, th, 255, cv2.THRESH_BINARY)[1]
    if hM==True:
        n_iter=1
        n_iter_bis=5
    else:
        n_iter=2
        n_iter_bis=4
    thresh = cv2.erode(thresh, None, iterations=n_iter)
    thresh = cv2.dilate(thresh, None, iterations=n_iter_bis)
    
    return thresh

def register_img(img_virtual,img_lowMag,th):
    img_virtual_bin=binarize_img(img_virtual,th,True)
    img_lowMag_bin=binarize_img(img_lowMag,th,False)

    transf, (pos_img_virtual, pos_img_lM) = find_transform(img_virtual_bin, img_lowMag_bin)
    registered_image = apply_transform(transf,img_virtual, img_lowMag)
    img_aligned=registered_image[0]*3

    
    return img_aligned,transf,pos_img_virtual,pos_img_lM


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


def run_registration(Folders,th):
    for filename in os.listdir(Folders['lowMag']):
        if 'fluorescent' in filename:
            print(filename)
            low_mag_id_1=int(filename[1:3])
            low_mag_id_2=int(filename[4:6])

            full_image_name=str(low_mag_id_1)+'_'+str(low_mag_id_2)+str('.png')
            img_virtual=cv2.imread(Folders['Virtual']+full_image_name,0)
            img_lowMag=cv2.imread(Folders['lowMag']+filename,0)
            try: 
                img_aligned,_,_,_=register_img(img_virtual,img_lowMag,th)
                cv2.imwrite(Folders['Aligned']+full_image_name,img_aligned)
                print("Registration Succesful")
            except MaxIterError: 
                print('ERROR: Could not find Registration -- skip FOV')
                continue
            except TooFewPointsError: 
                print('Too Few Points')
                continue
            except: #Try to understand what this is
                print('Other Error')
                continue
            

        else:
            continue
