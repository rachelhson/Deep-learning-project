# Overview 

## Project I : Generative Modeling of Complex Data Densities 
1. Modeling with single Norm distribution
2. Mixture of Gaussian 
3. T-distribution

## Project II : Adaptive Boosting (AdaBoost) for Face Detection
To boost the performance of the face detection, `Haar-Like feature` (weak learner) is used as an input of Adaboost classifier.





## Project III : Babysitting Training of Deep Neural Network (DNN)
Develop a neural network model using CIFAR-10 dataset, and tune parameters to improve the performance of image classification task. 
After model is trained and tuned with CIFAR-10 dataset, then the model is tested with the face dataset in `face-data` folder. After then, different architecture of model is also tested and compared.

hyperparmeter optmization : `learning rate`,`weight decay`,`momentum`,`layer change`
- learning rate : It controls how much to change the model in response to the estimated error each time the model weight are updated
- weight decay : | It updates loss value, applying it to the weights and bias (PyTorch). It avoids the weights/bias to avoid exploding gradient. 
- momentum |value: 0.0 ~ 1.0| It updates the step to find a new point in the search space (0.0 is the same as gradient decent w/o momentum)

* ROC curve (receiver operating characteristic curve) - a graph shows the performance of the model
- TPR (True Positive Rate) = TP / (TP + FN)
- FPR (False Positive Rate)= FP / (FP + TN)
