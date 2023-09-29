import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended_values = torch.sum(attention_weights * x, dim=1)
        return attended_values

class NeuralNetworkWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNetworkWithAttention, self).__init__()
        self.attention = Attention(input_dim, hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attended_values = self.attention(x)
        #output = self.fc2(torch.relu(self.fc1(attended_values)))
        pass 

def neural_network(X, y, stratification, additional_stratification, test_size=0.2, n_iterations=1):
    correlations = []

    for _ in range(n_iterations):
        X_train, X_test, y_train, y_test, strat_train, strat_test, _, _ = train_test_split(
            X, y, stratification, additional_stratification, test_size=test_size)

        # Convert data to PyTorch tensors
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
        X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)

        # Create and train the model
        input_dim = X_train.shape[1]
        hidden_dim = 64  # You can adjust this as needed
        model = NeuralNetworkWithAttention(input_dim, hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        for epoch in range(100):  # You can adjust the number of epochs
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train.view(-1, 1))
            loss.backward()
            optimizer.step()

            # Calculate validation loss
            with torch.no_grad():
                val_predictions = model(X_test)
                val_loss = criterion(val_predictions, y_test.view(-1, 1))

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

        # Make predictions on the test set
        with torch.no_grad():
            y_pred = model(X_test)

        # Calculate the correlation coefficient
        mse = mean_squared_error(y_test, y_pred)
        if mse > 0:
            correlation = np.sqrt(1 - mse)
        else:
            correlation = 0.0

        correlations.append(correlation)

        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Epochs vs Loss (Iteration {_ + 1})')
        plt.show()

    # Calculate the mean correlation coefficient over iterations
    mean_correlation = np.mean(correlations)

    return mean_correlation

import os
os.chdir("D:/Reaction optimization project/source code") 
from DFT.featurising_dataset import featurize_main_data




data_path ="D:/Reaction optimization project/source code/DFT/descriptor_data/"
dataset_path='D:/Reaction optimization project/source code/DATA/Dataset1.6.csv'

X_dft, y_dft, DOI_dft, mechanisms_dft, origins_dft = featurize_main_data(dataset_path,data_path)

a =  neural_network(X_dft, y_dft, DOI_dft,mechanisms_dft, n_iterations=5)

print(a)
