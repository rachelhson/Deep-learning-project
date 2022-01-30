# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:17:21 2020

@author: Rache
"""
from skimage import io
from sklearn.model_selection import train_test_split
import glob
import numpy as np
images =glob.glob("data/*.jpg")
def data ():
    data_arr = []
    for i in range(len(images)):
        image_arr = io.imread(images[i])/255 # to make [0,1]
        data_arr.append(image_arr)
    return data_arr

def data_split(image_data):
 #(1100_faces and 1100_nonfaces)
     X = image_data
     y = np.array([1]*1100 + [0]*1100)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.09090909)
     return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # data_set is the whole images 
    data_set = data()
    X_train, X_test,y_train,y_test = data_split(data_set)
    print(f'training set size {np.shape(X_train)}')
    print(f'testing set size {np.shape(X_test)}')
    
