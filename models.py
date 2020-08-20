
import torch
import torch.nn as nn
from   torch.autograd import Variable
##########################################################

#Linear regression model 
class linearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

##########################################################

#Deep neural network model 

class deepNN(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(inputSize, 128)
        self.bn1    = nn.BatchNorm1d(128)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(128, outputSize)
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.bn1(x)
        x = self.sigmoid(x)
        x = self.output(x)
        
        return x
class deepNN_1(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(inputSize, 128)
        self.bn1    = nn.BatchNorm1d(128)
        # Output layer, 10 units - one for each digit
        self.hidden2 = nn.Linear(128, 64)
        self.bn2    = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, outputSize)
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        x = self.output(x)
        
        return x
