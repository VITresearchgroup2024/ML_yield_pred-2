import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import copy



from skopt import BayesSearchCV

def random_forest(X, y, test_size=0.2, n_iterations=1):
    
    
    model_values =[]
    expt_values =[]
    # Iterate over test sets  
    for i in range(n_iterations):
        X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        
        # Train the model and get predictions 
        predictor = RandomForestRegressor(n_estimators=100)
        pred = copy.deepcopy(predictor)
        pred.fit(X_training, y_training)
        y_pred = pred.predict(X_test)
        
        model_values.extend(y_pred)
        expt_values.extend(y_test)
        
        return np.array(expt_values) ,np.array(model_values)
                                       
def random_forest_h_tuning_grid(X, y,
                                        test_size=0.2, n_iterations=1):
    '''
    a function that performs hyperparameter tuning for a Random Forest model to minimize
    mean absolute error (MAE) and returns the true values of y and model predictions 
    '''
    
    true_values = []
    model_values = []

    for i in range(n_iterations):
       
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size ,random_state=i)

        # Define the Random Forest Regressor
        rf = RandomForestRegressor()

        # Define the hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [10, 50, 100,150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2,4]
        }

        # Perform grid search hyperparameter tuning using MAE as the scoring metric
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   scoring='r2', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best Random Forest model from the grid search
        best_rf = grid_search.best_estimator_

        # Make predictions on the test set
        y_true = y_test
        y_pred = best_rf.predict(X_test)

        # Append true values and model predictions
        true_values.extend(y_true)
        model_values.extend(y_pred)

    return np.array(true_values), np.array(model_values)



def random_forest_h_tuning_bayes_strat(X, y,
                                        test_size=0.2, n_iterations=1):
    true_values = []
    model_values = []
    for i in n_iterations:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size ,random_state=i)
    
        # Define the Random Forest Regressor with hyperparameters to tune
        rf = RandomForestRegressor()
    
        # Define the search space for hyperparameter tuning
        search_space = {
            'n_estimators': (10, 1000),
            'max_depth': (1, 32),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 20)
        }
    
        # Perform Bayesian hyperparameter tuning using MAE as the objective function
        bayes_search = BayesSearchCV(
            rf, search_space, n_iter=30, cv=5, scoring='r2', n_jobs=-1
        )
        bayes_search.fit(X_train, y_train)
    
        # Get the best hyperparameters and their corresponding MAE
        best_params = bayes_search.best_params_
        best_mae = -bayes_search.best_score_
    
        # Train a Random Forest model with the best hyperparameters on the full training set
        best_rf = RandomForestRegressor(**best_params)
        best_rf.fit(X_train, y_train)
    
        # Make predictions on the test set
        y_true = y_test
        y_pred = best_rf.predict(X_test)
    
        # Append true values and model predictions
        true_values.extend(y_true)
        model_values.extend(y_pred)

    return np.array(true_values), np.array(model_values)
