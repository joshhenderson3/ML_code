import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import KFold  # <-- NEW IMPORT
from Models import LinearRegression
from Models import NeuralNetwork


class Trainer:
    """
    Class for training different ML models utilising different optimisation algorithms
    """

    def __init__(self, name, model_obj, eta, n_iter):
        """
        parameters
        ----------
        name (str): name of the optimiser used
        model_obj (obj): instance of Models class containing the ML model selected
        eta (float): learning rate, value in between 0 and 1
        n_iter (int): number of iterations
        """
        self.name = name
        self.model_obj = model_obj
        self.eta = eta
        self.n_iter = n_iter

    # Cross-Validation Method

    def cross_validate(
        self, X, y, n_splits=5
    ):  ###should this split be done outside the trainer class and instead in the input file??
        """
        Method that performs K-Fold Cross-Validation

        **Parameters**
        ----------
        X (torch tensor): All features (full dataset).
        y (torch tensor): All labels (full dataset).
        n_splits (int): Number of folds for CV (default is 5).

        Returns
        ------
        avg_loss (float): The average loss (MSE) across all folds.
        all_fold_losses (list): List of loss values for each fold.
        """
        # KFold ensures deterministic (reproducible) splits across CV runs/folds
        kf = KFold(
            n_splits=n_splits, shuffle=True, random_state=234
        )  # THIS SEED IS HARDCODED!
        all_fold_losses = []
        criterion = nn.MSELoss()

        # Convert PyTorch tensors to NumPy arrays for KFold to work easily
        X_np = X.detach().numpy()
        y_np = y.detach().numpy()

        # Identify Model Type
        ModelClass = self.model_obj.__class__  # Get the class of the model object
        architecture = None
        random_state = 0  # Set to Zero to prevent NoneType error

        if ModelClass.__name__ == "NeuralNetwork":
            # Assumes architecture is saved in Models.py
            architecture = self.model_obj.architecture
            random_state = self.model_obj.random_state

        elif ModelClass.__name__ == "LinearRegression":
            random_state = self.model_obj.seed

        print(f"Starting {n_splits}-Fold Cross-Validation...")

        for fold, (train_index, val_index) in enumerate(kf.split(X_np)):
            print(f"--- Fold {fold+1}/{n_splits} ---")

            # 1. Split data for current fold
            X_train_fold = torch.tensor(X_np[train_index], dtype=torch.double)
            y_train_fold = torch.tensor(y_np[train_index], dtype=torch.double)
            X_val_fold = torch.tensor(X_np[val_index], dtype=torch.double)
            y_val_fold = torch.tensor(y_np[val_index], dtype=torch.double)

            # 2. Initialise a new Model instance
            if architecture is not None:  # Neural Network Case
                new_model = ModelClass(
                    self.model_obj.name,
                    X_train_fold,
                    random_state + fold,  # Use a different seed per fold
                    architecture,
                )

            else:  # Linear Regression Case
                new_model = ModelClass(
                    self.model_obj.name,
                    X_train_fold,
                    random_state + fold,  # Use a different seed per fold
                )

                # 3. Initialise a new Trainer for the new model
            new_trainer = Trainer(self.name, new_model, self.eta, self.n_iter)
            new_trainer.training(X_train_fold, y_train_fold)
            # End of initialisation

            # 4. Evaluate the Model on the validation set
            with torch.no_grad():
                output = new_model.predict(X_val_fold)
                val_loss = criterion(output, y_val_fold).item()
            all_fold_losses.append(val_loss)
            print(f"Fold {fold+1} Validation Loss (MSE): {val_loss:.6f}")

        avg_loss = np.mean(all_fold_losses)
        print(f"\nAverage Cross-Validation Loss {n_splits} folds (MSE): {avg_loss:.6f}")
        return avg_loss, all_fold_losses

    # End of Cross-Validation Method

    def optim_selection(self):
        """
        Methods to choose which optimisation algorithm to use for the training
        """
        if self.name == "SGD":

            self.optimiser = optim.SGD(self.model_obj.train_parameters, self.eta)
            print("The optimiser is Stochastic Gradient Descent")

        elif self.name == "Adam":

            self.optimiser = optim.Adam(self.model_obj.train_parameters, self.eta)
            print("The optimiser is Adam")

        return self

    def training(self, X, y):
        """
        Method that train the ML model using the selected optimiser as implemented in PyTorch
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        y (torch tensor): dim = nx1 with n number of labeled points
        """

        self.losses = []  # losses array initialisation
        criterion = nn.MSELoss()  # loss function Mean Square Error

        self.optim_selection()
        # training routine
        for i in range(self.n_iter):

            output = self.model_obj.net_input(X)
            loss = criterion(output, y)
            self.optimiser.zero_grad()  # zeroing gradients
            loss.backward()  # calculate gradients
            self.optimiser.step()  # updating parameters
            self.losses.append(loss.item())
        return self

    def gd_optim(self, X, y):
        """
        Method for gradient descent training used by trainer2 for comparison
        """

        self.losses = []

        # Check if the model is Linear Regression
        if isinstance(self.model_obj, LinearRegression):

            # Use original weights and bias which exist in Linear Regression
            W = self.model_obj.weights
            B = self.model_obj.bias

            # Enable gradients once before the loop starts
            # This is critical for building the computation graph
            W.requires_grad_(True)
            B.requires_grad_(True)

            for i in range(self.n_iter):

                # Initialise gradients to zero before loop
                if W.grad is not None:
                    W.grad.zero_()
                if B.grad is not None:
                    B.grad.zero_()

                # 1. Forward Pass
                y_pred = self.model_obj.net_input(X)

                # 2. Compute loss
                loss = torch.mean((y_pred - y) ** 2)
                self.losses.append(loss.detach().numpy())

                # 3. Backward Pass
                loss.backward()

                # 4. Update Weights
                with torch.no_grad():
                    W -= self.eta * W.grad
                    B -= self.eta * B.grad

                if i % 10 == 0:
                    print(f"Iteration: {i:03d}, Loss: {loss.item():.6f}")

        else:
            # Neural Network Case

            # Use train_parameters, essential for NN
            for param in self.model_obj.train_parameters:
                param.requires_grad_(True)

            for i in range(self.n_iter):

                # 1. Zero out gradients before computing losses
                # Stops gradient accumulation
                for param in self.model_obj.train_parameters:
                    if param.grad is not None:
                        param.grad.zero_()

                # 2. Forward Pass
                y_pred = self.model_obj.net_input(X)

                # 3. Compute Loss
                loss = torch.mean((y_pred - y) ** 2)
                self.losses.append(loss.detach().numpy())

                # 4. Backward Pass
                loss.backward()

                # 5. Update Weights (GD Step)
                with torch.no_grad():
                    for param in self.model_obj.train_parameters:
                        param -= self.eta * param.grad

                if i % 10 == 0:  # print progress every 10 iterations
                    print(f"Iteration: {i:03d}, Loss: {loss.item():.6f}")

        return self

        # Does not work with Neural Network Implementation

        """
        Method that train the ML model using gradient descent as implemented in PyTorch
        Parameters
        ----------
        X (torch tensor): dim = nxm with n number of points (rows) and m number of features (columns)
        y (torch tensor): dim = nx1 with n number of labeled points
        """
        """
        self.model_obj.weights.requires_grad_(False)
        self.model_obj.bias.requires_grad_(False)
        self.losses = [] # losses array initialisation

        # training routine
        for i in range(self.n_iter):
            output = self.model_obj.net_input(X)
            errors = (y - output)
            self.model_obj.weights += self.eta * 2.0 * (torch.matmul(torch.t(X),errors)) / X.shape[0]
            self.model_obj.bias += self.eta * 2.0 * torch.mean(errors)
            errors_2 = torch.pow(errors,2)
            loss = torch.mean(errors_2)
            self.losses.append(loss)

        return self
        """
