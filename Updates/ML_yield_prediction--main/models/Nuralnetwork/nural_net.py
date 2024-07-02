import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Adjust the number of hidden units as needed
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer with 1 neuron for regression

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def neural_network(X, y, test_size, n_iterations,epochl,lr):
    """Train a neural network regressor using PyTorch and return the correlation coefficient.
    
    Parameters:
        X (np array): Features of the dataset, of shape (n_samples, n_features).
        y (np array): Labels of the dataset.
        stratification (np.array): Additional labels to use for the baseline.
        
        test_size (float): Test size (between 0 and 1) at which to perform the analysis.
        n_iterations (int): Number of iterations.

    Returns:
        cor_coeff (float): Correlation coefficient.
    """
    expt_values =[]
    model_values =[]

    for i in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,test_size=test_size, random_state=i
        )
        print(X_train)
        # Convert data to PyTorch tensors
        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train)
        X_test = torch.Tensor(X_test)
        y_test = torch.Tensor(y_test)
        
           
        # Create and train the neural network
        input_size = X_train.shape[1]
        model = NeuralNetwork(input_size)
        optimizer = optim.Adam(model.parameters(), lr) 
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        for epoch in range(epochl):  
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
        '''    # Calculate validation loss
            with torch.no_grad():
                val_predictions = model(X_test)
                val_loss = criterion(val_predictions, y_test.view(-1, 1))

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.title(f'Epochs vs Loss (Iteration {_ + 1})')
        plt.show()
        '''
        
        # Make predictions using the trained model
        y_pred = model(X_test)

        
        y_test = y_test.detach().numpy()
        y_pred = y_pred.detach().numpy()
        
        expt_values.extend(y_test)
        model_values.extend(y_pred) 
        
    new_model_val =[]             
    for x in model_values :
        x =str(x[0])
        new_model_val.append(x)
    return expt_values , new_model_val 
