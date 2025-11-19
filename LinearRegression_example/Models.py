import pandas as pd
import numpy as np
import torch
import torch.nn as nn # New import for neural networks
from sklearn.svm import SVR # New import for support vector regression
from sklearn.preprocessing import StandardScaler # For feature scaling


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
        self.seed = seed # Save seed for re-initialisation in cross validation
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

    def __init__(self, name, X, seed, architecture):
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

        # Save architecture and random state for cross validation re-initialisation
        self.architecture = architecture
        self.random_state = seed

        # Set random seed
        torch.manual_seed(seed)

        # Define layers based on architecture
        layers = []
        # Iterate through architecture to create layers
        for i in range(len(architecture) - 1):
            # The current layer output is the next layer input
            in_features = architecture[i]
            out_features = architecture[i + 1]

            # Create linear layer
            layers.append(nn.Linear(in_features, out_features, dtype = torch.double))

            # Add activation function (ReLU) except for the last layer
            if i < len(architecture) - 2:
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

class SupportVectorRegression(Models):
    '''
    Child class for Support Vector Regression (SVR) model
    Uses scikit-learn's SVR implementation
    '''

    def __init__ (self, name, X_train, y_train, seed, svr_params):
        '''
        Parameters
        ----------
        name (str): name of the model
        X_train (numpy array): training features
        Y_train (numpy array): training labels
        seed (int): seed for random number generator for reproducibility
        svr_params (dict): dictionary containing SVR parameters like C, kernel, gamma
        '''

        super().__init__(name) # inheriting all the Parent Class methods

        # Save parameters and seed for re-initialisation in cross validation
        self.svr_params = svr_params
        self.random_state = seed
        self.name = name

        # SVR requires requires data scalling
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        # Convert torch tensors to numpy arraus for scikit-learn compatibility
        X_np = X_train.numpy()
        y_np = y_train.numpy().flatten()

        # Scale the data and fit the scalers
        X_scaled = self.scaler_X.fit_transform(X_np)
        Y_scaled = self.scaler_Y.fit_transform(y_np.reshape(-1, 1)).flatten()

        # Initialise the SVR model
        self.model = SVR(
            C=svr_params['C'],
            kernel=svr_params['kernel'],
            gamma=svr_params['gamma']
        )

        # Fit the SVR model
        print("Fitting the SVR model...")
        self.model.fit(X_scaled, Y_scaled)
        print("SVR model fitted successfully.")

        # Crucial for compatability: no PyTorch parameters to track/optimise
        self.train_parameters = []

    def net_input(self,X):
        '''
        Method for calculating the scaled feature imput (NumPy array output)
        '''
        X_np = X.numpy()
        X_scaled = self.scaler_X.transform(X_np)
        return X_scaled
    
    def predict(self,X):
        '''
        Method for applying the SVR model in a predictive way
        Handles scaling, prediction and inverse scaling
        '''

        X_scaled = self.net_input(X)
        prediction_scaled = self.model.predict(X_scaled)

        # Inverse scale the predictions
        prediction_original = self.scaler_Y.inverse_transform(prediction_scaled.reshape(-1,1))

        # Convert back to torch tensor
        prediction_tensor = torch.from_numpy(prediction_original).double()

        return prediction_tensor




