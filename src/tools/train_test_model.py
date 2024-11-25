# Standard library imports
import time

# Third-party imports
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.neighbors import NearestCentroid, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor


def train_model(configuration, X_train, y_train, X_test, y_test):
    """
    Train the model with the given configuration and data.
    
    Parameters:
    ----------
    configuration : dict
        A dictionary containing the model type and its parameters.
    X_train : array-like
        Training data features.
    y_train : array-like
        Training data labels.
    X_test : array-like
        Testing data features.
    y_test : array-like
        Testing data labels.

    Returns:
    -------
    y_test : array-like
        True labels for the test set.
    y_pred : array-like
        Predicted labels for the test set.
    training_time : float
        Time taken to train the model.
    testing_time : float
        Time taken to test the model.
    """
    model_name = configuration.get('model_type')
    config = {key: value for key, value in configuration.items() if key != 'model_type'}

    # Initialize the model with the given configuration
    if model_name == 'svr':
        clf = svm.SVR(**config)
    elif model_name == 'decision_tree':
        clf = DecisionTreeRegressor(**config)
    elif model_name == 'random_forest':
        clf = RandomForestRegressor(**config)
    elif model_name == 'linear_regression':
        clf = LinearRegression(**config)
    elif model_name == 'knn':
        clf = KNeighborsRegressor(**config)
    elif model_name == 'mlp':
        clf = MLPRegressor(**config)
    elif model_name == 'gradient_boosting':
        clf = GradientBoostingRegressor(**config)
    elif model_name == 'lasso':
        clf = Lasso(**config)
    elif model_name == 'ridge':
        clf = Ridge(**config)
    elif model_name == 'elastic_net':
        clf = ElasticNet(**config)
    elif model_name == 'nearest_centroid':
        clf = NearestCentroid(**config)
    elif model_name == 'kmeans':
        clf = KMeans(**config)
    elif model_name == 'adaboost':
        clf = AdaBoostRegressor(**config)
    elif model_name == 'dummy_mean':
        clf = DummyRegressor(strategy='mean')
    elif model_name == 'dummy_median':
        clf = DummyRegressor(strategy='median')
    else:
        raise ValueError(f"Invalid model_type: {model_name}")

    # Train the model and measure the training time
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Test the model and measure the testing time
    start_time = time.time()
    y_pred = clf.predict(X_test)
    testing_time = time.time() - start_time

    return y_test, y_pred, training_time, testing_time