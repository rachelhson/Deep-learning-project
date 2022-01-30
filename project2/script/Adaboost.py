# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:03:38 2020

@author: Rache
"""

from read_data import data_split
from read_data import data 

import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
## importing data 
data_set = data()
X_train,X_test,y_train,y_test = data_split(data_set)

## pick the most impacting haar_feature 
haar_feature =['type-3-x','type-2-y']

def pixelsize(coord):
    for j in range(len(coord)):
        #dark side 
        dleft  =  coord[j][0][0]
        #print(f'left {dleft}')
        dx1  = dleft[0]
        #print(f'dx1 {dx1}')
        dy1  = dleft[1]
        #print(f'dy1 {dy1}')
        dright  =  coord[j][0][1]
        #print(f'dright {dright}')
        dx2  = dright[0]
        #print(f'dx2 {dx2}')
        dy2  = dright[1]
        #print(f'dy2 {dy2}')
        #lleft  = coord[1][0]
        #print(f'lleft {lleft}')
        #lx1  = lleft[0]
        #print(f'lx1 {lx1}')
        #ly1  = lleft[1]
        #print(f'ly1 {ly1}')
        #lright  =  coord[1][1]
        #print(f'lright {lright}')
        #lx2  = lright[0]
        #print(f'lx2 {lx2}')
        #ly2  = lright[1]
        #print(f'ly2 {ly2}')
        x_size =abs(dx1-dx2)+1
        #print(f'x_size{x_size}')
        y_size = abs(dy1-dy2)+1
        #print(f'y_size{y_size}')
        size = x_size*y_size
        #print(f'size{size}')
    return size
            

def weak_classifier(imagedata,lht):
    err = 1/len(imagedata)*np.cumsum(lht)
    return err

def f_lht(y,ht):
    lhf_arr = []
    for i in range(len(ht)):
        if y[i] != ht[i] : 
            lht = 1
            lhf_arr.append(lht)
        else: 
            lht = 0
            lhf_arr.append(lht)
    return lhf_arr 

def f_alpha(et):
    alphat = 1/2*np.log((1-et)/et)
    return alphat 


if __name__ == "__main__":
    feature_a =[]
    
    for i in range(len(X_train)):
         ii_img = integral_image(X_train[i])
        
         feature = haar_like_feature(ii_img, 0, 0, ii_img.shape[0], ii_img.shape[1],
                             feature_type=haar_feature)
         coord, feature_type = \
         haar_like_feature_coord(20,20, feature_type = haar_feature)
         px = pixelsize(coord)
         feature_a.append(feature)
    
    f_s = [] # models
    lhf_arr = []
    
    
   #evaluation
    accuracy = []
    predictions = []
    err_arr = []
    Xf = feature_a
    Xf_train,Xf_test,yf_train,yf_test =train_test_split(Xf,y_train,test_size=0.090)
    T = 10
    w = 1/len(Xf_train) * np.ones(len(Xf_train))
    for t in range(T):
         
        # use DecisionTreeClassifier to find weak classifier 
        dlf = DecisionTreeClassifier(max_depth=1)
        #feature
        f = dlf.fit(Xf_train, yf_train, sample_weight = w)
        f_s.append(f)
        # prediction
        prediction = dlf.predict(Xf_train)
        
        # score 
        s = f.score(Xf_train, yf_train)
        #Evaluation misclassified (misclassified 1, and classified 0 )
        lhf_arr = f_lht(yf_train,prediction)
        
        # sum of error of all images
        err = np.sum(w*lhf_arr)/np.sum(w)
        print(f'error is {err}')
        err_arr.append(err)
        alphat = f_alpha(err)
        print(f'alpha is {alphat}')
        alphat_arr = np.ones(len(lhf_arr))* alphat
        if [lht == 1 for lht in lhf_arr]:
            alphat_arr = -1 * np.ones(len(alphat_arr))*alphat_arr
            s =np.sum(np.exp(alphat_arr*lhf_arr)*w)
        else: 
            s =np.sum(np.exp(alphat_arr*lhf_arr)*w)
       
        w = w * np.exp(alphat_arr*lhf_arr)/s
        print(f'boosted weight is {w}')
        #fig = plt.figure(figsize=(5,5))
        #ax0 = fig.add_subplot()
        idx_sorted = np.argsort(dlf.feature_importances_)[::-1]
        print(f'idx{idx_sorted}')
        
        prediction = dlf.predict(Xf_test)
        score = dlf.score(Xf_test,yf_test)
        print(f'score{score}')
        predictions.append(prediction)
        fig2,ax2 = plt.subplots()
        ns_probs =[0 for _ in range(len(yf_test))]
        lr_probs = dlf.predict_proba(Xf_test)
        lr_probs = lr_probs[:,1]
        ns_auc = roc_auc_score(yf_test,ns_probs)
        lr_auc = roc_auc_score(yf_test,lr_probs)
        ns_fpr, ns_tpr, _ = roc_curve(yf_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(yf_test, lr_probs)
        ax2.plot(ns_fpr, ns_tpr, linestyle='--')
        ax2.plot(lr_fpr, lr_tpr, marker='.')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        weighted_roc_auc_ovo = roc_auc_score(yf_test, lr_probs, multi_class="ovo",
                                     average="weighted")
        print(f'weighted_roc_auc_ovo{weighted_roc_auc_ovo}')
        plt.show()
        auc_full_features = roc_auc_score(yf_test, dlf.predict_proba(Xf_test)[:, 1])
        print(auc_full_features)
        
    
    #print(f'prediction{predictions}')
    
        if [yf for yf in yf_test] == [p for p in prediction]: 
            predictions.append(1)
        accuracy.append((np.sum(predictions))/len(predictions[0]))
            #print(accuracy
    fig3,ax3 = plt.subplots()
    ax3.plot(range(len(accuracy)),accuracy,'-b')
    ax3.set_xlabel('# models used for Boosting ')
    ax3.set_ylabel('accuracy')
            #print('With a number of ',T,'base models we receive an accuracy of ',accuracy[-1]*100,'%')    
    plt.show()        
        

       
        #ax0.plot(t,w[0])
        #plt.show()
    fig,axes = plt.subplots(3,6)
    for idx, ax in enumerate(axes.ravel()):
        imag = X_train[100]
        image = draw_haar_like_feature(imag, 0, 0,20,20,[coord[idx_sorted[idx]]])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
    _ = fig.suptitle('The most important features')
    plt.axis('off')
    plt.show()
  
            
  

    
   