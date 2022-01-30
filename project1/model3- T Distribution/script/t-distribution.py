# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:53:27 2020

@author: Rache
"""
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from scipy import special
from scipy.optimize import fsolve
import math
import glob
from single_ import FR, Label, ROC, SingleGaussian


resolution = 20

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

##FaceTestData
test_face_images=glob.glob("resized_face_test/*.jpg")
face_test=[]
for i in range(100):
    im = cv2.imread(test_face_images[i])
    x = np.reshape(im[:,:,:],-1)
    face_test.append(x)
Face_Test_images = np.array(face_test)

##NonFaceTestData
test_nonface_images=glob.glob("resized_nonface_test/*.jpg")
nonface_test=[]
for i in range(100):
    im = cv2.imread(test_nonface_images[i])
    x = np.reshape(im[:,:,:],-1)
    nonface_test.append(x)
Nonface_Test_images = np.array(nonface_test)


##concentanate data in train and test 
Train_images = np.zeros((2000,20*20*3))
Train_images[0:1000,]=Face_Train_images
Train_images[1000:2000,]=Nonface_Train_images
Test_images= np.zeros((200,20*20*3))
Test_images[0:100,] = Face_Test_images
Test_images[100:200,] = Nonface_Test_images


def T_dist(data,v=3):
    
    (N,D)=data.shape
    mu=np.mean(data,axis=0)
    sigma=np.cov(data.transpose())
 
    
    for i in range(30):
        center = data - mu
        temp=v+np.sum(np.multiply(np.dot(center,np.linalg.inv(sigma)),center),axis=1)
        Exp_h=(v+D)/temp#N
        Exp_log_h=special.digamma((v+D)/2)-np.log(temp/2)
        
        #M step
        mu=np.sum(np.multiply(data.transpose(),Exp_h),axis=1)/np.sum(Exp_h)
        temp_center_current=data-mu
        sigma=np.dot(np.multiply(temp_center_current.transpose(),Exp_h),temp_center_current)/N
        def f(v):
            return(np.log(v/2)+1-special.digamma(v/2)+np.mean(Exp_log_h-Exp_h))
        v=fsolve(f,v)

    return[mu,sigma,v]
              
 
#Evaluate the learned model on the testing images
Test_true_label=np.zeros(200)
Test_true_label[0:100]=1 
Train_true_label=np.zeros(2000)
Train_true_label[0:1000]=1 

print(FR(Train_images,true_label=Train_true_label,threshold=0.5,v_start=5))
print(FR(Test_images,true_label=Test_true_label,threshold=0.5,v_start=5))
ROC(Test_images,true_label=Test_true_label,ratio_threshold_seq=np.arange(-1500,1500,100),v_start=5)

#mean
[Face_u,Face_sigma,Face_v]=EM_T(Face_Train_images,v_start=3)
[Nonface_u,Nonface_sigma,Nonface_v]=EM_T(Nonface_Train_images,v_start=3)

plt.subplot(2, 2, 1)
plt.imshow(Face_u.reshape((20,20,3)).astype(int))
plt.title("mean-Face")

plt.subplot(2, 2, 2)
plt.imshow(Nonface_u.reshape((20,20,3)).astype(int))
plt.title("mean-NonFace")

#cov
plt.subplot(2, 2, 3)
cov_diag=np.diag(Face_sigma)    
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((20,20,3)).astype(int))
plt.title("cov-Face")

plt.subplot(2, 2, 4)
cov_diag=np.diag(Nonface_sigma)  
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((20,20,3)).astype(int))
plt.title("cov-NonFace")