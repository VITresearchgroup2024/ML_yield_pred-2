import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


'''
 function that builds a neural network model with attention, performs hyperparameter tuning to
 minimize mean absolute error (MAE), and considers stratified sampling, you can use libraries like 
 scikit-learn, pytorch, and skopt for Bayesian optimization
 '''
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
        output = self.fc2(torch.relu(self.fc1(attended_values)))
        return output

def neural_network_with_attention_hyperparameter_tuning(X, y,
                                                        test_size=0.2, n_iterations=1):
    true_values = []
    model_values = []

    for _ in range(n_iterations):
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=None)

        # Convert data to PyTorch tensors
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
        X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)

        # Define the hyperparameter search space
        search_space = {
            'hidden_dim': Integer(16, 128),
            'lr': Real(0.001, 0.1),
            'batch_size': Integer(32, 256),
            'epochs': Integer(10, 100),
        }

        # Define the objective function to minimize MAE
        def objective_function(hidden_dim, lr, batch_size, epochs):
            model = NeuralNetworkWithAttention(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.L1Loss()  # Mean Absolute Error (L1 Loss)

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(batch_X)
                    loss = criterion(y_pred, batch_y.view(-1, 1))
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            return mae

        # Perform Bayesian hyperparameter tuning using MAE as the objective function
        bayes_search = BayesSearchCV(
            objective_function, search_space, n_iter=30, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_absolute_error'
        )
        bayes_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = bayes_search.best_params_

        # Train the final model with the best hyperparameters
        best_model = NeuralNetworkWithAttention(input_dim=X_train.shape[1], hidden_dim=best_params['hidden_dim'])
        optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'])
        criterion = nn.L1Loss()  # Mean Absolute Error (L1 Loss)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

        for epoch in range(best_params['epochs']):
            best_model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = best_model(batch_X)
                loss = criterion(y_pred, batch_y.view(-1, 1))
                loss.backward()
                optimizer.step()

        best_model.eval()
        with torch.no_grad():
            y_true = y_test
            y_pred = best_model(X_test)

        # Append true values and model predictions
        true_values.extend(y_true.numpy())
        model_values

