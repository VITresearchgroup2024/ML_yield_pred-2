import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score




def preprocess_yield(y):
    # Preprocess the target variable y into four classes
    classes = []
    for yi in y:
        if yi < 25:
            classes.append("POOR YIELD")
        elif yi < 50:
            classes.append("BELOW AVERAGE YIELD")
        elif yi < 75:
            classes.append("AVERAGE YIELD")
        else:
            classes.append("GOOD YIELD")
    return np.array(classes)

def knn_classification_HPT(X, y,test_size=0.2, n_iterations=10):
    true_values =[]
    model_values =[]
    y=preprocess_yield(y)

    for i in range(n_iterations):
        
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=i)

        knn = KNeighborsClassifier()

        # Define the hyperparameter search space
        param_grid = {
            'n_neighbors': [3, 5, 7,9],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],# Number of neighbors to consider
            'weights': ['uniform', 'distance'],  # Weighting scheme
            'p': [1, 2]  # Distance metric (1 for Manhattan, 2 for Euclidean)
        }

        # Perform GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid,scoring='accuracy', cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_

        # Train KNN Classifier with the best hyperparameters on the full training set
        best_knn = KNeighborsClassifier(**best_params)
        best_knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = best_knn.predict(X_test)

    true_values.extend(y_test)
    model_values.extend(y_pred)

    return np.array(true_values), np.array(model_values)




def knn_classification(X, y,test_size=0.2, n_iterations=5):
    true_values =[]
    model_values =[]
    y=preprocess_yield(y)
    for i in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=i)

     
        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)  

        # Train KNN Classifier on the full training set
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

    true_values.extend(y_test)
    model_values.extend(y_pred)

    return np.array(true_values), np.array(model_values)

    
 

