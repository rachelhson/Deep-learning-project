# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 02:34:18 2020

@author: Rache
"""
import pandas as pd
from sklearn.mixture import GaussianMixture 
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
##FaceTrainData 
train_face_images=glob.glob("resized_face/*.jpg")
face_train=[]
for i in range(1000):
    im = cv2.imread(train_face_images[i])
    x=np.reshape(im[:,:,:],-1)
    face_train.append(x)
Face_Train_images = np.array(face_train)
##NonFaceTrainData
train_nonface_images=glob.glob("resized_nonface/*.jpg")
nonface_train=[]
for i in range(1000):
    im = cv2.imread(train_nonface_images[i])
    x=np.reshape(im[:,:,:],-1)
    nonface_train.append(x)
Nonface_Train_images= np.array(nonface_train)

d = pd.DataFrame(Face_Train_images)


plt.scatter (d[0],d[1])
gmm = GaussianMixture(n_components=3)
gmm.fit(d)
# Assign a label to each sample 
labels = gmm.predict(d) 
d['labels']= labels 
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2] 
# plot three clusters in same plot 
plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 

# print the converged log-likelihood value 
print(gmm.lower_bound_) 
  
# print the number of iterations needed 
# for the log-likelihood value to converge 
print(gmm.n_iter_)

train_nonface_images=glob.glob("resized_nonface/*.jpg")
nonface_train=[]
for i in range(1000):
    im = cv2.imread(train_nonface_images[i])
    x=np.reshape(im[:,:,:],-1)
    nonface_train.append(x)
Nonface_Train_images= np.array(nonface_train)



d = pd.DataFrame(Nonface_Train_images)
plt.scatter (d[0],d[1])
gmm = GaussianMixture(n_components=3)
gmm.fit(d)
# Assign a label to each sample 
labels = gmm.predict(d) 
d['labels']= labels 
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2] 
# plot three clusters in same plot 
plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 

# print the converged log-likelihood value 
print(gmm.lower_bound_) 
  
# print the number of iterations needed 
# for the log-likelihood value to converge 
print(gmm.n_iter_)
