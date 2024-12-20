configurations_dummy = [
        {'model_type': 'dummy_mean', 'strategy': 'mean'},
        {'model_type': 'dummy_median', 'strategy': 'median'},  
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.01},
        {'model_type': 'svr', 'kernel': 'rbf', 'C': 1, 'gamma': 0.01},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'distance', 'p': 2},
        {'model_type':'ridge', 'alpha': 4},
        {'model_type': 'lasso', 'alpha': 0.2},
        {'model_type': 'ridge', 'alpha': 1000},
        {'model_type': 'ridge', 'alpha': 500}, 
        {'model_type': 'ridge', 'alpha': 125},
        {'model_type': 'lasso', 'alpha': 0.1},
        {'model_type': 'kmeans', 'n_clusters': 10, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type': 'linear_regression', 'fit_intercept': True},
        {'model_type': 'svr', 'kernel': 'sigmoid', 'C': 0.001, 'gamma': 10, 'coef0': 3},
        {'model_type': 'knn', 'n_neighbors': 3, 'weights': 'uniform', 'p': 2},
        {'model_type': 'svr', 'kernel': 'poly', 'C': 0.1, 'degree': 5, 'coef0': 3},
        {'model_type': 'knn', 'n_neighbors': 3, 'weights': 'distance', 'p': 2},
        {'model_type': 'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.001},
              
]
configurations = [
        {'model_type': 'dummy_mean', 'strategy': 'mean'},
        {'model_type': 'dummy_median', 'strategy': 'median'},  
    
        # SVR configurations
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.0001},
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.001},
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.01},
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.1},
        {'model_type':'svr', 'kernel': 'linear', 'C': 1},
        {'model_type':'svr', 'kernel': 'linear', 'C': 50},
        {'model_type':'svr', 'kernel': 'poly', 'C': 0.001, 'degree': 5, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'poly', 'C': 0.1, 'degree': 5, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'poly', 'C': 1, 'degree': 5, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 5, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 6, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 7, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 5, 'coef0': 4},
        {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 5, 'coef0': 5},
        {'model_type':'svr', 'kernel': 'poly', 'C': 100, 'degree': 5, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'poly', 'C': 1000, 'degree': 5, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 0.001, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 0.01, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 0.1, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 1, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.001},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.1},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 1},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 28, 'gamma': 6},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 100, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 1000, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.0095},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.015},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 5, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 5, 'gamma': 0.0095},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 5, 'gamma': 0.015},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 50, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 50, 'gamma': 0.0095},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 50, 'gamma': 0.015},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 0.001, 'gamma': 10, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 0.01, 'gamma': 10, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 0.1, 'gamma': 10, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 1, 'gamma': 10, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 0.010, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 0.10, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 1, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 100, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 1},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 2},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 4},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 5},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 100, 'gamma': 10, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'sigmoid', 'C': 1000, 'gamma': 10, 'coef0': 3},

        
        # decision_tree configurations
        {'model_type':'decision_tree', 'max_depth': None, 'min_samples_split': 5},
        {'model_type':'decision_tree', 'max_depth': 1, 'min_samples_split': 7},
        {'model_type':'decision_tree', 'max_depth': 2, 'min_samples_split': 7},
        {'model_type':'decision_tree', 'max_depth': 3, 'min_samples_split': 7},
        {'model_type':'decision_tree', 'max_depth': 2, 'min_samples_split': 10},
        {'model_type':'decision_tree', 'max_depth': 2, 'min_samples_split': 15},
        
        # # random_forest configurations
        # {'model_type':'random_forest', 'n_estimators': 100, 'max_depth': None},
        # {'model_type':'random_forest', 'n_estimators': 50, 'max_depth': 5},
        # {'model_type':'random_forest', 'n_estimators': 100, 'max_depth': 5},
        # {'model_type':'random_forest', 'n_estimators': 100, 'max_depth': 10},
        # {'model_type':'random_forest', 'n_estimators': 200, 'max_depth': 5},
        
        # # knn configurations
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'distance', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'distance', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'distance', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'distance', 'p': 1},
        
        # # linear_regression configurations
        {'model_type':'linear_regression', 'fit_intercept': True},
        {'model_type':'linear_regression', 'fit_intercept': False},
        
        # # MLP configurations
        # {'model_type':'mlp', 'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (150,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (50,), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (100,), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (150,), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (50,), 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (100,), 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (150,), 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},

        # # Gradient Boosting configurations
        # {'model_type':'gradient_boosting', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        # {'model_type':'gradient_boosting', 'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 1},
        # {'model_type':'gradient_boosting', 'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 2},
        # {'model_type':'gradient_boosting', 'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3},
        # {'model_type':'gradient_boosting', 'n_estimators': 150, 'learning_rate': 0.2, 'max_depth': 2},
        # {'model_type':'gradient_boosting', 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 1},
        # {'model_type':'gradient_boosting', 'n_estimators': 100, 'learning_rate': 0.01, 'max_depth': 3},
        # {'model_type':'gradient_boosting', 'n_estimators': 150, 'learning_rate': 0.01, 'max_depth': 3},
        # {'model_type':'gradient_boosting', 'n_estimators': 150, 'learning_rate': 0.01, 'max_depth': 2},
        # {'model_type':'gradient_boosting', 'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 3},

        # # Lasso configurations
        {'model_type':'lasso', 'alpha': 0.01},
        {'model_type':'lasso', 'alpha': 0.1},
        {'model_type':'lasso', 'alpha': 0.2},
        {'model_type':'lasso', 'alpha': 0.4},
        {'model_type':'lasso', 'alpha': 0.5},
        {'model_type':'lasso', 'alpha': 0.6},
        {'model_type':'lasso', 'alpha': 0.7},
        {'model_type':'lasso', 'alpha': 0.8},
        {'model_type':'lasso', 'alpha': 0.9},

        # # Ridge configurations
        {'model_type':'ridge', 'alpha': 0.1},
        {'model_type':'ridge', 'alpha': 0.2},
        {'model_type':'ridge', 'alpha': 0.3},
        {'model_type':'ridge', 'alpha': 0.4},
        {'model_type':'ridge', 'alpha': 0.5},
        {'model_type':'ridge', 'alpha': 0.6},
        {'model_type':'ridge', 'alpha': 0.7},
        {'model_type':'ridge', 'alpha': 0.8},
        {'model_type':'ridge', 'alpha': 2},
        {'model_type':'ridge', 'alpha': 4},
        {'model_type':'ridge', 'alpha': 7},
        {'model_type':'ridge', 'alpha': 15},
        {'model_type':'ridge', 'alpha': 30},
        {'model_type':'ridge', 'alpha': 60},
        {'model_type':'ridge', 'alpha': 125},
        {'model_type':'ridge', 'alpha': 250},
        {'model_type':'ridge', 'alpha': 500},
        {'model_type':'ridge', 'alpha': 1000},
        {'model_type':'ridge', 'alpha': 1500},


        # # Elastic Net configurations
        # {'model_type':'elastic_net', 'alpha': 0.1, 'l1_ratio': 0.3},
        # {'model_type':'elastic_net', 'alpha': 0.01, 'l1_ratio': 0.05},
        # {'model_type':'elastic_net', 'alpha': 0.1, 'l1_ratio': 0.7},
        {'model_type':'elastic_net', 'alpha': 0.5, 'l1_ratio': 0.3},
        # {'model_type':'elastic_net', 'alpha': 0.5, 'l1_ratio': 0.5},
        # {'model_type':'elastic_net', 'alpha': 0.5, 'l1_ratio': 0.7},
        # {'model_type':'elastic_net', 'alpha': 1.0, 'l1_ratio': 0.3},
        # {'model_type':'elastic_net', 'alpha': 1.0, 'l1_ratio': 0.5},
        # {'model_type':'elastic_net', 'alpha': 1.0, 'l1_ratio': 0.7},

        # # Nearest Centroid configurations
        # # {'model_type':'nearest_centroid', 'metric': 'euclidean', 'shrink_threshold': None},
        # # {'model_type':'nearest_centroid', 'metric': 'euclidean', 'shrink_threshold': 0.1},
        # # {'model_type':'nearest_centroid', 'metric': 'euclidean', 'shrink_threshold': 0.2},
        # # {'model_type':'nearest_centroid', 'metric': 'manhattan', 'shrink_threshold': None},
        # # {'model_type':'nearest_centroid', 'metric': 'manhattan', 'shrink_threshold': 0.1},
        # # {'model_type':'nearest_centroid', 'metric': 'manhattan', 'shrink_threshold': 0.2},

        # # KMeans configurations
        {'model_type':'kmeans', 'n_clusters': 2, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type':'kmeans', 'n_clusters': 3, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type':'kmeans', 'n_clusters': 4, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type':'kmeans', 'n_clusters': 5, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type':'kmeans', 'n_clusters': 6, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type':'kmeans', 'n_clusters': 7, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type':'kmeans', 'n_clusters': 8, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type':'kmeans', 'n_clusters': 9, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type':'kmeans', 'n_clusters': 10, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},

        # # AdaBoost configurations
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 0.5, 'loss': 'linear'},
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 0.5, 'loss': 'square'},
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 0.5, 'loss': 'exponential'},
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 1.0, 'loss': 'linear'},
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 1.0, 'loss': 'square'},
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 1.0, 'loss': 'exponential'},
        # {'model_type':'adaboost', 'n_estimators': 100, 'learning_rate': 0.5, 'loss': 'linear'},
        # {'model_type':'adaboost', 'n_estimators': 100, 'learning_rate': 0.5, 'loss': 'square'},
        # {'model_type':'adaboost', 'n_estimators': 100, 'learning_rate': 0.5, 'loss': 'exponential'}
    ]
configurations_optimized = [
        # SVR configurations
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.0001},
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.001},
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.01},
        # {'model_type':'svr', 'kernel': 'linear', 'C': 0.1},
        # {'model_type':'svr', 'kernel': 'linear', 'C': 1},
        # {'model_type':'svr', 'kernel': 'linear', 'C': 50},
        {'model_type':'svr', 'kernel': 'poly', 'C': 0.001, 'degree': 5, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 0.1, 'degree': 5, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 1, 'degree': 5, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 5, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 6, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 7, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 5, 'coef0': 4},
        {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 5, 'coef0': 5},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 100, 'degree': 5, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 1000, 'degree': 5, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 0.001, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 0.01, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 0.1, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 1, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.001},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.1},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 1},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 28, 'gamma': 6},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 100, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 1000, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.0095},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.015},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 5, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 5, 'gamma': 0.0095},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 5, 'gamma': 0.015},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 50, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 50, 'gamma': 0.0095},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 50, 'gamma': 0.015},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 0.001, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 0.01, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 0.1, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 1, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 0.010, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 0.10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 1, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 100, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 1},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 2},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 4},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 5},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 100, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 1000, 'gamma': 10, 'coef0': 3},

        # DummyRegressor configurations
        {'model_type': 'dummy_mean', 'strategy': 'mean'},
        {'model_type': 'dummy_median', 'strategy': 'median'},        
        
        # decision_tree configurations
        # {'model_type':'decision_tree', 'max_depth': None, 'min_samples_split': 5},
        # {'model_type':'decision_tree', 'max_depth': 1, 'min_samples_split': 7},
        {'model_type':'decision_tree', 'max_depth': 2, 'min_samples_split': 7},
        {'model_type':'decision_tree', 'max_depth': 3, 'min_samples_split': 7},
        # {'model_type':'decision_tree', 'max_depth': 2, 'min_samples_split': 10},
        # {'model_type':'decision_tree', 'max_depth': 2, 'min_samples_split': 15},
        
        # random_forest configurations
        {'model_type':'random_forest', 'n_estimators': 100, 'max_depth': None},
        # {'model_type':'random_forest', 'n_estimators': 50, 'max_depth': 5},
        # {'model_type':'random_forest', 'n_estimators': 100, 'max_depth': 5},
        {'model_type':'random_forest', 'n_estimators': 100, 'max_depth': 10},
        # {'model_type':'random_forest', 'n_estimators': 200, 'max_depth': 5},
        
        # knn configurations
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'distance', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'distance', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'distance', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'distance', 'p': 1},
        
        # linear_regression configurations
        # {'model_type':'linear_regression', 'fit_intercept': True},
        # {'model_type':'linear_regression', 'fit_intercept': False},
        
        # MLP configurations
        # {'model_type':'mlp', 'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (150,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (50,), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (100,), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (150,), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        {'model_type':'mlp', 'hidden_layer_sizes': (50,), 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        {'model_type':'mlp', 'hidden_layer_sizes': (100,), 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},
        # {'model_type':'mlp', 'hidden_layer_sizes': (150,), 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 10000, 'learning_rate_init': 0.001, 'early_stopping': True},

        # Gradient Boosting configurations
        # {'model_type':'gradient_boosting', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        # {'model_type':'gradient_boosting', 'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 1},
        # {'model_type':'gradient_boosting', 'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 2},
        # {'model_type':'gradient_boosting', 'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3},
        # {'model_type':'gradient_boosting', 'n_estimators': 150, 'learning_rate': 0.2, 'max_depth': 2},
        # {'model_type':'gradient_boosting', 'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 1},
        # {'model_type':'gradient_boosting', 'n_estimators': 100, 'learning_rate': 0.01, 'max_depth': 3},
        {'model_type':'gradient_boosting', 'n_estimators': 150, 'learning_rate': 0.01, 'max_depth': 3},
        {'model_type':'gradient_boosting', 'n_estimators': 150, 'learning_rate': 0.01, 'max_depth': 2},
        # {'model_type':'gradient_boosting', 'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 3},

        # Lasso configurations
        # {'model_type':'lasso', 'alpha': 0.01},
        {'model_type':'lasso', 'alpha': 0.1},
        {'model_type':'lasso', 'alpha': 0.2},
        # {'model_type':'lasso', 'alpha': 0.4},
        # {'model_type':'lasso', 'alpha': 0.5},
        # {'model_type':'lasso', 'alpha': 0.6},
        # {'model_type':'lasso', 'alpha': 0.7},
        # {'model_type':'lasso', 'alpha': 0.8},
        # {'model_type':'lasso', 'alpha': 0.9},

        # Ridge configurations
        # {'model_type':'ridge', 'alpha': 0.1},
        # {'model_type':'ridge', 'alpha': 0.2},
        # {'model_type':'ridge', 'alpha': 0.3},
        # {'model_type':'ridge', 'alpha': 0.4},
        # {'model_type':'ridge', 'alpha': 0.5},
        # {'model_type':'ridge', 'alpha': 0.6},
        # {'model_type':'ridge', 'alpha': 0.7},
        # {'model_type':'ridge', 'alpha': 0.8},
        # {'model_type':'ridge', 'alpha': 2},
        # {'model_type':'ridge', 'alpha': 4},
        
        # {'model_type':'ridge', 'alpha': 7},
        # {'model_type':'ridge', 'alpha': 15},
        # {'model_type':'ridge', 'alpha': 30},
        # {'model_type':'ridge', 'alpha': 60},
        # {'model_type':'ridge', 'alpha': 125},
        # {'model_type':'ridge', 'alpha': 250},
        {'model_type':'ridge', 'alpha': 500},
        {'model_type':'ridge', 'alpha': 1000},
        {'model_type':'ridge', 'alpha': 1500},


        # Elastic Net configurations
        {'model_type':'elastic_net', 'alpha': 0.1, 'l1_ratio': 0.3},
        # {'model_type':'elastic_net', 'alpha': 0.01, 'l1_ratio': 0.05},
        {'model_type':'elastic_net', 'alpha': 0.1, 'l1_ratio': 0.7},
        # {'model_type':'elastic_net', 'alpha': 0.5, 'l1_ratio': 0.3},
        # {'model_type':'elastic_net', 'alpha': 0.5, 'l1_ratio': 0.5},
        # {'model_type':'elastic_net', 'alpha': 0.5, 'l1_ratio': 0.7},
        # {'model_type':'elastic_net', 'alpha': 1.0, 'l1_ratio': 0.3},
        # {'model_type':'elastic_net', 'alpha': 1.0, 'l1_ratio': 0.5},
        # {'model_type':'elastic_net', 'alpha': 1.0, 'l1_ratio': 0.7},

        # Nearest Centroid configurations
        # {'model_type':'nearest_centroid', 'metric': 'euclidean', 'shrink_threshold': None},
        # {'model_type':'nearest_centroid', 'metric': 'euclidean', 'shrink_threshold': 0.1},
        # {'model_type':'nearest_centroid', 'metric': 'euclidean', 'shrink_threshold': 0.2},
        # {'model_type':'nearest_centroid', 'metric': 'manhattan', 'shrink_threshold': None},
        # {'model_type':'nearest_centroid', 'metric': 'manhattan', 'shrink_threshold': 0.1},
        # {'model_type':'nearest_centroid', 'metric': 'manhattan', 'shrink_threshold': 0.2},

        # KMeans configurations
        {'model_type':'kmeans', 'n_clusters': 2, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type':'kmeans', 'n_clusters': 3, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 4, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 5, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 6, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 7, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 8, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 9, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 10, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},

        # AdaBoost configurations
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 0.5, 'loss': 'linear'},
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 0.5, 'loss': 'square'},
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 0.5, 'loss': 'exponential'},
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 1.0, 'loss': 'linear'},
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 1.0, 'loss': 'square'},
        # {'model_type':'adaboost', 'n_estimators': 50, 'learning_rate': 1.0, 'loss': 'exponential'},
        {'model_type':'adaboost', 'n_estimators': 100, 'learning_rate': 0.5, 'loss': 'linear'},
        # {'model_type':'adaboost', 'n_estimators': 100, 'learning_rate': 0.5, 'loss': 'square'},
        {'model_type':'adaboost', 'n_estimators': 100, 'learning_rate': 0.5, 'loss': 'exponential'}
    ]

configurations_waves_stats = [
        # SVR configurations
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.0001},
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.001},
        {'model_type':'svr', 'kernel': 'linear', 'C': 0.01},
        # {'model_type':'svr', 'kernel': 'linear', 'C': 0.1},
        # {'model_type':'svr', 'kernel': 'linear', 'C': 1},
        # {'model_type':'svr', 'kernel': 'linear', 'C': 50},
        {'model_type':'svr', 'kernel': 'poly', 'C': 0.001, 'degree': 5, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'poly', 'C': 0.1, 'degree': 5, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 1, 'degree': 5, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 5, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 6, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 7, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 5, 'coef0': 4},
        {'model_type':'svr', 'kernel': 'poly', 'C': 10, 'degree': 5, 'coef0': 5},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 100, 'degree': 5, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'poly', 'C': 1000, 'degree': 5, 'coef0': 3},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 0.001, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 0.01, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 0.1, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 1, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.001},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.1},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 1},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 28, 'gamma': 6},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 100, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 1000, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.0095},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 10, 'gamma': 0.015},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 5, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 5, 'gamma': 0.0095},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 5, 'gamma': 0.015},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 50, 'gamma': 0.01},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 50, 'gamma': 0.0095},
        {'model_type':'svr', 'kernel': 'rbf', 'C': 50, 'gamma': 0.015},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 0.001, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 0.01, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 0.1, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 1, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 0.010, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 0.10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 1, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 100, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 1},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 2},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 4},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 10, 'gamma': 10, 'coef0': 5},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 100, 'gamma': 10, 'coef0': 3},
        # {'model_type':'svr', 'kernel': 'sigmoid', 'C': 1000, 'gamma': 10, 'coef0': 3},

        
        # decision_tree configurations
        # {'model_type':'decision_tree', 'max_depth': None, 'min_samples_split': 5},
        # {'model_type':'decision_tree', 'max_depth': 1, 'min_samples_split': 7},
        {'model_type':'decision_tree', 'max_depth': 2, 'min_samples_split': 7},
        {'model_type':'decision_tree', 'max_depth': 3, 'min_samples_split': 7},
        # {'model_type':'decision_tree', 'max_depth': 2, 'min_samples_split': 10},
        # {'model_type':'decision_tree', 'max_depth': 2, 'min_samples_split': 15},
        

        
        # knn configurations
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 3, 'weights': 'distance', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 5, 'weights': 'distance', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'uniform', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'distance', 'p': 2},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 7, 'weights': 'distance', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'uniform', 'p': 1},
        {'model_type':'knn', 'n_neighbors': 10, 'weights': 'distance', 'p': 1},
        
        # linear_regression configurations
        {'model_type':'linear_regression', 'fit_intercept': True},
        {'model_type':'linear_regression', 'fit_intercept': False},
        
     
     
        # Lasso configurations
        # {'model_type':'lasso', 'alpha': 0.01},
        {'model_type':'lasso', 'alpha': 0.1},
        {'model_type':'lasso', 'alpha': 0.2},
        # {'model_type':'lasso', 'alpha': 0.4},
        # {'model_type':'lasso', 'alpha': 0.5},
        # {'model_type':'lasso', 'alpha': 0.6},
        # {'model_type':'lasso', 'alpha': 0.7},
        # {'model_type':'lasso', 'alpha': 0.8},
        # {'model_type':'lasso', 'alpha': 0.9},

        # Ridge configurations
        # {'model_type':'ridge', 'alpha': 0.1},
        # {'model_type':'ridge', 'alpha': 0.2},
        # {'model_type':'ridge', 'alpha': 0.3},
        # {'model_type':'ridge', 'alpha': 0.4},
        # {'model_type':'ridge', 'alpha': 0.5},
        # {'model_type':'ridge', 'alpha': 0.6},
        # {'model_type':'ridge', 'alpha': 0.7},
        # {'model_type':'ridge', 'alpha': 0.8},
        # {'model_type':'ridge', 'alpha': 2},
        # {'model_type':'ridge', 'alpha': 4},
        
        # {'model_type':'ridge', 'alpha': 7},
        # {'model_type':'ridge', 'alpha': 15},
        # {'model_type':'ridge', 'alpha': 30},
        # {'model_type':'ridge', 'alpha': 60},
        # {'model_type':'ridge', 'alpha': 125},
        # {'model_type':'ridge', 'alpha': 250},
        {'model_type':'ridge', 'alpha': 500},
        {'model_type':'ridge', 'alpha': 1000},
        {'model_type':'ridge', 'alpha': 1500},


        # Elastic Net configurations
        {'model_type':'elastic_net', 'alpha': 0.1, 'l1_ratio': 0.3},
        # {'model_type':'elastic_net', 'alpha': 0.01, 'l1_ratio': 0.05},
        {'model_type':'elastic_net', 'alpha': 0.1, 'l1_ratio': 0.7},
        # {'model_type':'elastic_net', 'alpha': 0.5, 'l1_ratio': 0.3},
        # {'model_type':'elastic_net', 'alpha': 0.5, 'l1_ratio': 0.5},
        # {'model_type':'elastic_net', 'alpha': 0.5, 'l1_ratio': 0.7},
        # {'model_type':'elastic_net', 'alpha': 1.0, 'l1_ratio': 0.3},
        # {'model_type':'elastic_net', 'alpha': 1.0, 'l1_ratio': 0.5},
        # {'model_type':'elastic_net', 'alpha': 1.0, 'l1_ratio': 0.7},

        # Nearest Centroid configurations
        # {'model_type':'nearest_centroid', 'metric': 'euclidean', 'shrink_threshold': None},
        # {'model_type':'nearest_centroid', 'metric': 'euclidean', 'shrink_threshold': 0.1},
        # {'model_type':'nearest_centroid', 'metric': 'euclidean', 'shrink_threshold': 0.2},
        # {'model_type':'nearest_centroid', 'metric': 'manhattan', 'shrink_threshold': None},
        # {'model_type':'nearest_centroid', 'metric': 'manhattan', 'shrink_threshold': 0.1},
        # {'model_type':'nearest_centroid', 'metric': 'manhattan', 'shrink_threshold': 0.2},

        # KMeans configurations
        {'model_type':'kmeans', 'n_clusters': 2, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        {'model_type':'kmeans', 'n_clusters': 3, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 4, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 5, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 6, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 7, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 8, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 9, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
        # {'model_type':'kmeans', 'n_clusters': 10, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},

    ]

