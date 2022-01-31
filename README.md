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

1. Activation fuinction : `ReLu`(Rectified Linear Unit) function. The range of ReLu is [0,inf]. 

2. Define the network : 
```
def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2) # (kernel size, stride)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connected layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```
3. hyperparmeter optmization : `learning rate`,`weight decay`,`momentum`,`layer change`
- `learning rate` : It controls how much to change the model in response to the estimated error each time the model weight are updated. Too small learning rate results in taking long time for training, while too large value cause learning too fast or unstable training process.

- `weight decay (Regularization)` : | It updates loss value, applying it to the weights and bias (PyTorch). It avoids the weights/bias to avoid exploding gradient. 
- `momentum` |value: 0.0 ~ 1.0| It updates the step to find a new point in the search space (0.0 is the same as gradient decent w/o momentum)
- `the number of neurons, activation function, optimizer, learning rate, batch size, and epochs`
* ROC curve (receiver operating characteristic curve) - a graph shows the performance of the model
- TPR (True Positive Rate) = TP / (TP + FN)
- FPR (False Positive Rate)= FP / (FP + TN)
