import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import os

import tifffile as tiff

from astroalign import *

def flip_rotate_images(Folders):
    for filename in os.listdir(Folders['HighMag']):
        if filename.endswith(".tif"): 
            print(filename)
            #read only the first channel, which corresponds to DAPI
            img=tiff.imread(Folders['HighMag']+filename)
            img=img[1,:,:]
            img=img.transpose()
            img=cv2.flip(img,0)
            img=cv2.flip(img,1)
            nbrs=filename[-11:-4]
            nb1=int(nbrs[0:3])
            nb2=int(nbrs[4:8])
            new_name=str(nb1)+'_'+str(nb2)+'.tif'
            cv2.imwrite(Folders['HighMag_RF']+new_name,img)
        else:
            continue



def concatenate_crop_ROI(nb_pixel_x, nb_pixel_y,delta,fov_oct,Folders,offset):
    #concatenate FOVs, then crop ROI

    pixel_size=0.365623*10**-6
    size_cropping=fov_oct+2*delta
    n_pixels_map=round(size_cropping/pixel_size)
    fov_nh_x=nb_pixel_x*pixel_size#nh = nighthawk, i.e. high mag
    fov_nh_y=nb_pixel_y*pixel_size

    for filename in os.listdir(Folders['lowMag']):
        if 'fluorescent' in filename:
            
            print(filename)
            low_mag_id_1=int(filename[1:3])
            low_mag_id_2=int(filename[4:6])
            y_start=max(low_mag_id_1*fov_oct-delta,0)+offset['y']
            x_start=max(low_mag_id_2*fov_oct-delta,0)+offset['x']
            y_end=y_start+fov_oct+2*delta
            x_end=x_start+fov_oct+2*delta
            
            #loop through the frames
            #identify which ones are the limit ones
            nb_im_x=18

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
    #         full_image=np.zeros((n_pixels_map,n_pixels_map))
            full_image=np.zeros((nb_pixel_y*(frame_y_end-frame_y_start+1),nb_pixel_x*(frame_x_end-frame_x_start+1)))
            #pad picture with corresponding frames
            idx_x=np.arange(frame_x_end,frame_x_start-1,-1)
            idx_y=np.arange(frame_y_end,frame_y_start-1,-1)

            for i in range(idx_x.shape[0]):
                for j in range(idx_y.shape[0]):
                    filename=str(idx_y[j])+'_'+str(idx_x[i])+'.tif'
                    if os.path.exists(Folders['HighMag_RF']+filename):
                        crt_img=cv2.imread(Folders['HighMag_RF']+filename,0)
                    else:
                        continue
                    full_image[j*nb_pixel_y:(j+1)*nb_pixel_y,i*nb_pixel_x:(i+1)*nb_pixel_x]=crt_img
                    
                    if i==0 and j==0:
                        x_start_crt=nb_pixel_x*(idx_x[i]-1)*pixel_size
                        y_start_crt=nb_pixel_y*(idx_y[j]-1)*pixel_size
                        x_ref=x_start_crt+nb_pixel_x*pixel_size
                        y_ref=y_start_crt+nb_pixel_y*pixel_size
                        x_idx=int(round((x_ref-x_end)/pixel_size))
                        y_idx=int(round((y_ref-y_end)/pixel_size))
            
            full_image=full_image[y_idx:y_idx+n_pixels_map,x_idx:x_idx+n_pixels_map]
                    
            
            full_image_name=str(low_mag_id_1)+'_'+str(low_mag_id_2)+str('.png')
            cv2.imwrite(Folders['Virtual']+full_image_name,full_image)
            print(full_image_name)
            print(full_image.shape)
            print("---------------")
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

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
                img_aligned,transf,pos_img_virtual,pos_img_lM=register_img(img_virtual,img_lowMag,th)
                cv2.imwrite(Folders['Aligned']+full_image_name,img_aligned)
                print("Registration Succesful")
            except MaxIterError: 
                print('ERROR: Could not find Registration -- skip FOV')
                continue
            except: 
                print('Other Error')
                continue
            

        else:
            continue
