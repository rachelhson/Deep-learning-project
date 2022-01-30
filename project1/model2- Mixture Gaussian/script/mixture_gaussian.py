# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 22:41:00 2020

@author: Rache
"""


import numpy as np
import matplotlib.pyplot as plt
from single_ import FR, Label, ROC, SingleGaussian, Log_p
import glob
import cv2
from scipy.stats import multivariate_normal


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

def random_mu_sigma(data,k):
    z = np.zeros((k,1200))
    lamda = np.zeros((k,))
    collect_k =[]
    for i in range(1000):
        z[:, i] = np.random.random((k, ))
        z[:, i] /= np.sum(z[:, i])  
        z_sum = np.sum(z) 
        for j in range(k):
            lamda[j]=np.sum(z[j,:])/z_sum
            [mu,sigma]=SingleGaussian(z[j,:])
            collect_k.append([mu,sigma])
            log_pp = np.zeros((0,1200))
            log_ = multivariate_normal.pdf(data[i],mu,sigma)
            log_pp = np.vstack((log_pp,log_))
        z = np.zeros(k,1000)
        for i in range(1000):
            p =np.multiply(np.exp(log_pp[:,i]-np.max(log_pp[:,i])))
            z[:,i]=p/np.sum(p)
  
def Gaussian_Mix(data,k=3): 
    
    (N,D)=data.shape
   
   
    mu=np.ones((k,D))
    sigma=np.ones((k,D,D))
    
    a=np.random.random((k, ))
    group_size=int(N/K)
    for i in range(K):
        mu[i,:]=np.mean(data[a[(group_size*i):(group_size*(i+1))],:],axis=0)
        sigma[i,:,:]=np.diag(np.diag(np.cov(data[a[(group_size*i):(group_size*(i+1))],].transpose())))
 
   
    
def E_step (data, k=3):
    (N,D)=data.shape
    mu=np.ones((k,D))
    lamda=np.zeros(k)*(1/k)
    log_p=np.ones((N,K))
    S=np.ones((N,K))
    sigma=np.ones((k,D,D))
    
    for t in range(30):
     
        for i in range(K):
            center=data-mu[k,:]
            log_p[:,k]=-(1/2)*np.sum(np.multiply(np.dot(center,np.linalg.pinv(sigma[k,:,:])),center),axis=1)-\
            (1/2)*np.sum(np.log(np.linalg.svd(sigma[k,:,:])[1]))
        for i in range(K):
            for n in range(N):
                S[n,k]=lamda[k]/np.sum(lamda*np.exp(Log_p[n,:]-Log_p[n,k]))
                mu=(np.dot(S.transpose(),input).transpose()/np.sum(S,axis=0)).transpose()#K*D
 
        for i in range(K):
           temp_center=data-mu[k,:]
           sigma[k,:,:]=np.dot(np.multiply(temp_center.transpose(),S[:,k]),temp_center)/np.sum(S[:,k])
     
        lamda=np.sum(S,axis=0)/np.sum(S)
        
  
      
        
        for i in range(k):
           sigma[k,:,:]=np.diag(np.diag(sigma[k,:,:]))
        
   
    return [mu,sigma,lamda]


#Evaluate the learned model on the testing images
Test_true_label=np.zeros(200)
Test_true_label[0:100]=1    
Train_true_label=np.zeros(2000)
Train_true_label[0:1000]=1

for K in range(3):
    print(K)
    print(FR(Train_images,Train_true_label,K=K,threshold=0.5))
    
for K in range(3):
    print(K)
    print(FR(Test_images,Test_true_label,K=K,threshold=0.5))
    print(FR(Test_images,Test_true_label,K=6,threshold=0.5))

(Face_sigma,Face_h,Face_u)=Gaussian_Mix(Face_Train_images,K=5)   
(Nonface_sigma,Nonface_h,Nonface_u)=Gaussian_Mix(Nonface_Train_images,K=5) 

ROC(Test_images,true_label=Test_true_label,ratio_threshold_seq=np.arange(-1500,1500,100),K=3)
ROC(Test_images,true_label=Test_true_label,ratio_threshold_seq=np.arange(-1500,1500,100),K=3)


plt.subplot(2, 2, 1)
#mean
(Face_sigma,Face_h,Face_u)=Gaussian_Mix(Face_Train_images,K=K)   
(Nonface_sigma,Nonface_h,Nonface_u)=Gaussian_Mix(Nonface_Train_images,K=K) 

plt.imshow(np.dot(Face_h,Face_u).reshape((20,20,3)).astype(int))
plt.title("mean-Face")

plt.subplot(2, 2, 2)
plt.imshow(np.dot(Nonface_h,Nonface_u).reshape((20,20,3)).astype(int))
plt.title("mean-NonFace")

plt.subplot(2, 2, 3)
#cov
cov_diag=np.zeros(20*20*3)
for i in range(Face_sigma.shape[0]):
    cov_diag=cov_diag+np.diag(Face_sigma[i,:,:])*Face_h[i]
    
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((20,20,3)).astype(int))
plt.title("cov-Face")

plt.subplot(2, 2, 4)
cov_diag=np.zeros(20*20*3)
for i in range(Face_sigma.shape[0]):
    cov_diag=cov_diag+np.diag(Nonface_sigma[i,:,:])*Nonface_h[i]
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((20,20,3)).astype(int))
plt.title("cov-NonFace")