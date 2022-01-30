# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:08:09 2020

@author: Rache
"""
import numpy as np
from skimage import io
from skimage.color import rgb2gray

import glob
from skimage.transform import resize
import matplotlib.pyplot as plt
import random

def facedataset ():
    
    face_images=glob.glob("data/*.jpg")
    offset = (50,50)
    #resolution =20  
    n = 100/20
    face_x =[]
##Facedata 
    for i in range (len(face_images)):
        color = io.imread(face_images[i])
        image = rgb2gray(color)
    #face_train_image_arr = np.zeros((1000,resolution,resolution))
    ## crop face setup
        center = (int (image.shape[0]/2), int (image.shape[1]/2))
        x_croprange_min = center[0]-offset[0]
        x_croprange_max = center[0]+offset[0]
        y_croprange_min = center[1]-offset[1]
        y_croprange_max = center[1]+offset[1]

        #coords =(x_croprange_min, y_croprange_min, x_croprange_max, y_croprange_max)
        cropped_image=image[x_croprange_min:x_croprange_max,y_croprange_min:y_croprange_max]
        #print(np.shape(cropped_image))
        ## resize image to resolution 
        image_resized = resize(cropped_image, (cropped_image.shape[0] //n , cropped_image.shape[1] // n),
                       anti_aliasing=True)
    
        #face_resized_image =cropped_image.resize((resolution,resolution))
        #face_resized_image_arr= np.array(face_resized_image)
        face_x.append(image_resized)
        #io.imsave ("C:\\Users\\Rache\\Documents\\GitHub\\ECE 763\\hson_project02\\00_pseudo\\data\\" + str(i) + ".jpg",image_resized)
    return face_x 



def nonfacedataset():    
## nonface data
    nonface_images=glob.glob("data/*.jpg")
    face_offset = (30,30)
    offset= 10
    #resolution =20
    n = 20/20
    nonface_x =[]

    for i in range (len(nonface_images)):
        color = io.imread(nonface_images[i])
        image = rgb2gray(color)

        ## crop face setup
        #center = (int(image.shape[0]/2),int(image.shape[1]/2))
        nonface_x1= random.uniform(image.shape[0]/2+face_offset[0]+offset, image.shape[0]-offset)

        cropped_image=image[int(nonface_x1-offset):int(nonface_x1+offset),int(nonface_x1-offset):int(nonface_x1+offset)]
        image_resized = resize(cropped_image, (cropped_image.shape[0] //n , cropped_image.shape[1] // n),
                       anti_aliasing=True)
   
        nonface_x.append(image_resized)
        #io.imsave ("C:\\Users\\Rache\\Documents\\GitHub\\ECE 763\\hson_project02\\00_pseudo\\data\\" + str(2000+i) + ".jpg",image_resized)
           
    return nonface_x
    
def data():
    face_x = facedataset()
    nonface_x = nonfacedataset()
    X = np.concatenate((face_x,nonface_x))
    y = np.array([1]*len(face_x)+[0]*len(nonface_x))
    print(f"alldatasize is {np.shape(X)}")
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(face_x[1000], cmap=plt.cm.Greys_r)
    ax[1].imshow(nonface_x[1000], cmap=plt.cm.Greys_r)
    plt.show()
    return X, y    

if __name__ == "__main__":
    
    data()
        
      