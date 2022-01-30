# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:30:15 2020

@author: Rache
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import glob

import cv2
import random

from scipy.stats import multivariate_normal
from scipy.stats import norm


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

def SingleGaussian(data):
    mu=np.mean(data,axis=0)
    sigma = np.cov(data.transpose())
    return [mu,sigma]

def Log_p(data,Face=True):
    if(Face==True):#for Face
        [mu,sigma]=SingleGaussian(Face_Train_images)
    else:#for Nonface
        [mu,sigma]=SingleGaussian(Nonface_Train_images)
    log_p = norm.logpdf(data,mu,sigma[0])
    #log_p=np.sum(np.log(np.linalg.svd(sigma)[1]))*(-1/2)-(1/2)*np.sum(np.multiply(np.dot(temp_center,np.linalg.pinv(sigma)),temp_center),axis=1)
    return log_p

def Log_p_facefinal(data,Face=True):
    ## face probability 
    log_p_face=Log_p(data,Face=True)
    log_p_nonface =Log_p(data,Face=False)
    
    log_p_facefinal = log_p_face/(log_p_face+log_p_nonface)
    return log_p_facefinal

def Log_p_nonfacefinal(data,Face=True):
    ## face probability 
    log_p_face=Log_p(data,Face=True)
    log_p_nonface =Log_p(data,Face=False)
    
    log_p_nonfacefinal = log_p_nonface/(log_p_face+log_p_nonface)
    return log_p_nonfacefinal

def result_f(data): 
    score_f=[]
    log_p_facefinal = Log_p_facefinal(data,Face=True)
    if log_p_facefinal[log_p_facefinal>0.5]:
        score_f.append(log_p_facefinal)
    return score_f

def Rate(data,true_label,threshold=0.5):
    n=data.shape[0]
    predict_score =Label(data)
    FR=np.zeros(3)
    FR[0]=np.mean(predict_score[[i for i in range(n) if true_label[i]==0]])
    FR[1]=1-np.mean(predict_score[[i for i in range(n) if true_label[i]==1]])
    FR[2]=np.mean(np.abs(predict_score-true_label))
    return FR

def ROC(data,true_label,ratio_threshold_seq):
    N=data.shape[0]
    delta=Log_p(data,Face=True)-Log_p(data,Face=False)#log_p_face-log_p_nonface
    if(isinstance(ratio_threshold_seq,np.ndarray)):#threshold is a seq
        FR=np.zeros((2,len(ratio_threshold_seq)))#false positive rate and false negative rate
        for i in range(len(ratio_threshold_seq)):
            #face_or_nonface
            ratio_threshold=ratio_threshold_seq[i]
            estimated_label=np.zeros(N)
            estimated_label[[i for i in range(N) if any(delta[i])>ratio_threshold]]=1
            #False Rate
            FR[0,i]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
            FR[1,i]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        plt.plot(FR[0,:],1-FR[1,:],"r--")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title("ROC-Gaussian")
        plt.show()
        
        
def Label(data,threshold=0.5):
    delta=Log_p(data,Face=True)-Log_p(data,Face=False)
    ratio_threshold=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):
        estimated_label=np.zeros(data.shape[0])
        estimated_label[[i for i in range(data.shape[0]) if any(delta[i])>ratio_threshold]]=1
    return(estimated_label)
    

def FR(Input_data,true_label,threshold=0.5):
    N=Input_data.shape[0]
    delta=Log_p(Input_data,Face=True)-Log_p(Input_data,Face=False)#log_p_face-log_p_nonface
    ratio_threshold=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):#threshold is a scalar
        #face_or_nonface
        estimated_label=np.zeros(N)
        estimated_label[[i for i in range(N) if delta[i]>ratio_threshold]]=1
        #False Rate
        FR=np.zeros(3)
        FR[0]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
        FR[1]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        FR[2]=np.mean(np.abs(estimated_label-true_label))
        return FR
                

def ROC_Gaussian(Input_data,true_label,ratio_threshold_seq):
    N=Input_data.shape[0]
    delta=Log_p(Input_data,Face=True)-Log_p(Input_data,Face=False)#log_p_face-log_p_nonface
    if(isinstance(ratio_threshold_seq,np.ndarray)):#threshold is a seq
        FR=np.zeros((2,len(ratio_threshold_seq)))#false positive rate and false negative rate
        for i in range(len(ratio_threshold_seq)):
            #face_or_nonface
            ratio_threshold=ratio_threshold_seq[i]
            estimated_label=np.zeros(N)
            estimated_label[[i for i in range(N) if delta[i]>ratio_threshold]]=1
            #False Rate
            FR[0,i]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
            FR[1,i]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        plt.plot(FR[0,:],1-FR[1,:],"r--")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title("ROC-Gaussian")
        plt.show()
 
#Evaluate the learned model on the testing images
Test_true_label=np.zeros(200)
Test_true_label[0:100]=1 
Train_true_label=np.zeros(2000)  
Train_true_label[0:1000]=1 

#EM_Gaussian(Face_Train_trans)
print(FR(Train_images,true_label=Train_true_label,threshold=0.5))
print(FR(Test_images,true_label=Test_true_label,threshold=0.5))
ROC_Gaussian(Test_images,true_label=Test_true_label,ratio_threshold_seq=np.arange(-1500,1500,100))


plt.subplot(2, 2, 1)
#mean
[Face_mu,Face_sigma]=SingleGaussian(Face_Train_images)
plt.imshow(Face_mu.reshape((20,20,3)).astype(int))
plt.title("mean-Face")
plt.subplot(2, 2, 2)
[Nonface_mu,Nonface_sigma]=SingleGaussian(Nonface_Train_images)
plt.imshow(Nonface_mu.reshape((20,20,3)).astype(int))
plt.title("mean-NonFace")

plt.subplot(2, 2, 3)
#cov
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
