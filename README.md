# Machine Learning

The code here was written to fit a machine learning model to a data for image detections.  
Three different models have been tested: linear regression, deepNN, and deepNN_1. (models.py)
The highest scoring one was deepNN_1. In this model I used two hidden layers, as well as a sigmoid function and batch normalization.
I used Adam as the optimizer with learning rate of 0.01 and amsgrad option in pytorch module.
