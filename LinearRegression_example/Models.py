import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class Models:
    '''
    Parent Class for different ML models. 
    It contains two methods meant to be overriden:
    net_input and predict
    '''

    def __init__(self,name):
        
        self.name = name
        print(f'Model choosen: {self.name}')

    def net_input(self):
        pass

    def predict(self):
        pass

class LinearRegression(Models):
    '''
    Child Class for a linear regression model 
    '''
    
    def __init__(self, name, X, seed):
        '''
        Parameters
        ----------
        name (str): name of the model 
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        seed (int): seed for random number generator for initial values of weights and bias
        '''
        super().__init__(name) # inheriting all the Parent Class methods
        torch.manual_seed(seed) # setting the random number generator with the seed for reproducibility

        self.weights = torch.rand((X.shape[1],1),requires_grad=True, dtype=torch.double) # initial values of weights w
        self.bias = torch.rand((1,1),requires_grad=True, dtype=torch.double) # initial value of the coefficient b
        self.train_parameters = [self.weights,self.bias]
    
    def net_input(self,X):
        '''
        Method for calculating the input for the training or for a more complicated ML model (ANN)
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        '''

        return torch.matmul(X,self.weights) + self.bias
    
    def predict(self,X):
        '''
        Method for applying the ML model in a predictive way
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        '''
        
        prediction = self.net_input(X)
        return prediction
    
# Linear Regression remains unchanged

class NeuralNetwork(Models, nn.Module):
    '''
    Child class for a feed-forward nerual network model 
    '''

    def __init__(self, name, X, seed,architeture):
        '''
        Parameters
        ----------
        name (str): name of the model
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        seed (int): seed for random number generator for reproducibility
        architecture (list): list containing the number of neurons in each layer (given in input.yml)
        '''

        # Initialise parent classes
        Models.__init__(self, name)
        # Initialise nn.Module
        nn.Module.__init__(self)

        # Set random seed
        torch.manual_seed(seed)

        # Define layers based on architecture
        layers = []
        # Iterate through architecture to create layers
        for i in range(len(architeture) - 1):
            # The current layer output is the next layer input
            in_features = architeture[i]
            out_features = architeture[i + 1]

            # Create linear layer
            layers.append(nn.Linear(in_features, out_features, dtype = torch.double))

            # Add activation function (ReLU) except for the last layer
            if i < len(architeture) - 2:
                layers.append(nn.ReLU())

        # Combine layers into a sequential model
        self.network = nn.Sequential(*layers)

        # The parameters list is required for the trainer class (weights/bias for each layer)
        self.train_parameters = list(self.network.parameters())

    def net_input(self,X):
        '''
        Method for calculating the forward pass through the neural network
        '''

        # Call the network on input tensor X
        return self.network(X)
    
    def predict(self,X):
        '''
        Method for applying the neural network in a predictive way
        '''

        # X (tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        prediction = self.net_input(X)
        return prediction
    
    def forward(self, X):
        '''
        Standard PyTorch method for the forward pass, calls the predict method
        '''
        return self.predict(X)


