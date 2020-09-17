# Fitting a deep neural network algorithm to a data set including a collection of textual descriptions of neural network model architectures trained on Cifar-10

The code here has been developed to work with a data-set,
representing the errors of different machine learning models
trained for image detection on Cifar-10. Each model has more than
200 features which are used as inputs in a python code, and two sets
of labels. The code then uses a deep neural network consists of
two hidden layers, a sigmoid function, and a batch normalization.
The model was fitted to the data-set using Adam as the optimizer
with learning rate of 0.01 and amsgrad option in Pytorch module.
