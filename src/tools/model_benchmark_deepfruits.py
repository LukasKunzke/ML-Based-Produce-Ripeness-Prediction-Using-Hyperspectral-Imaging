# Standard library imports
import time
import math
from collections import defaultdict

# Third-party imports
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
import pandas as pd
from collections import defaultdict

# Local application imports
from loader_deepfruit import load_spectral_data
from configurations import *
from train_test_model import train_model

def train_model_config(config, X_train, y_train, X_test, y_test):
    # Initialize lists to store the metrics for each configuration
    maes, mses, r2s, training_times, testing_times = [], [], [], [], []

    # Train the model and get the metrics
    y_test, y_pred, training_time, testing_time = train_model(config, X_train, y_train, X_test, y_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Append the metrics to the corresponding lists
    maes.append(mae)
    mses.append(mse)
    r2s.append(r2)
    training_times.append(training_time)
    testing_times.append(testing_time)

    # Calculate the average metrics
    avg_mae = np.mean(maes)
    avg_mse = np.mean(mses)
    avg_r2 = np.mean(r2s)
    avg_training_time = np.mean(training_times)
    avg_testing_time = np.mean(testing_times)
    
    return avg_mae, avg_mse, avg_r2, avg_training_time, avg_testing_time, y_pred
    

def evaluate_configurations(train_data, train_labels, test_data, test_labels, configurations):
    
    results = defaultdict(list)

    # Iterate over all configurations
    for config in configurations:
        avg_mae, avg_mse, avg_r2, avg_training_time, avg_testing_time, y_pred = train_model_config(config, train_data, train_labels, test_data, test_labels)
        # Store the average metrics and the configuration in the results dictionary
        results[(avg_mae, avg_mse, avg_r2)].append((config, {'Average Training Time': avg_training_time, 'Average Testing Time': avg_testing_time}))

    return results, y_pred

    
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    plt.show()


def compare_accuracy(config, train_data, train_labels, test_data, test_labels, y_category):
    print(config)
    results = defaultdict(list)
    avg_mae, avg_mse, avg_r2, avg_training_time, avg_testing_time, y_pred = train_model_config(config, train_data, train_labels, test_data, test_labels)
    results[(avg_mae, avg_mse, avg_r2)].append((config, {'Average Training Time': avg_training_time, 'Average Testing Time': avg_testing_time}))
    print(results)
    print(f"Predicted labels: {y_pred}")
    # Convert y_pred to integer categories
    y_pred_int = [1 if y <= 1.8 else 2 if y <= 2.25 else 3 for y in y_pred]
    
    # Convert lists to numpy arrays for consistent formatting
    y_pred_int = np.array(y_pred_int)
    y_category = np.array(y_category)
   
    plot_confusion_matrix(y_category, y_pred_int)
    print (f"Predicted labels: {y_pred_int}")
    print (f"Actual ripeness : {y_category}")
    
    accuracy = accuracy_score(y_category, y_pred_int)
    print(f"Accuracy: {accuracy}")

def print_results(results):
    """
    Print the evaluation results.
    
    Parameters:
    ----------
    results : dict
        Dictionary with configurations as keys and metrics as values.
    """
    print()
    for metrics, configs in results:
        print(f"Average Mean Absolute Error: {round(metrics[0], 4)}")
        print(f"Average Mean Squared Error : {round(metrics[1], 4)}")
        print(f"Average R^2 score          : {round(metrics[2], 4)}")
        for config, times in configs:
            print(f"Configuration: {config}")
            print(f"Average Training Time      : {round(times['Average Training Time'], 4)}")
            print(f"Average Testing Time       : {round(times['Average Testing Time'], 4)}")
        print()


def get_performance(results):
    """
    Get the performance metric from the results.
    
    Parameters:
    ----------
    results : dict
        Dictionary with configurations as keys and metrics as values.

    Returns:
    -------
    float
        The lowest value of the sorted category.
    """
    if not results:
        raise ValueError("No results to evaluate.")
    return results[-1][0][0]


def process_spectral_data(train_data, train_labels, test_data, test_labels, configurations):
    """
    Process the spectral data by evaluating configurations.
    
    Parameters:
    ----------
    all_data : array-like
        The dataset features.
    all_labels : array-like
        The dataset labels.
    random_states : list
        List of random states for reproducibility.
    configurations : list
        List of model configurations to evaluate.
    """
    results, y_pred = evaluate_configurations(train_data, train_labels, test_data, test_labels, configurations)
    sorted_results = sorted(results.items(), key=lambda item: item[0][0], reverse=True)
    print_results(sorted_results)

def remove_features(data, current_feature, features_to_eliminate):
    features_to_remove = features_to_eliminate + [current_feature]
    reduced_data = []

    for picture in data:
        reduced_picture = [wavelength for i, wavelength in enumerate(picture) if i not in features_to_remove]
        reduced_data.append(reduced_picture)

    return reduced_data

def feature_optimizer(train_data, train_labels, test_data, test_labels, configurations):
    feature_count = len(train_data[0])
    initial_results, _ = evaluate_configurations(train_data, train_labels, test_data, test_labels, configurations)
    sorted_results = sorted(initial_results.items(), key=lambda item: item[0][0], reverse=True)
    initial_performance = get_performance(sorted_results)
    best_result = (initial_performance, initial_results)
    features_to_eliminate = []

    for i in range(feature_count):
        reduced_train_data = remove_features(train_data, i, features_to_eliminate)
        reduced_test_data = remove_features(test_data, i, features_to_eliminate)
        reduced_results, _ = evaluate_configurations(reduced_train_data, train_labels, reduced_test_data, test_labels, configurations)
        sorted_results = sorted(reduced_results.items(), key=lambda item: item[0][0], reverse=True)
        reduced_performance = get_performance(sorted_results)

        if reduced_performance <= best_result[0]:
            features_to_eliminate.append(i)
            best_result = (reduced_performance, sorted_results)

    print("Features to be eliminated:", features_to_eliminate)
    print_results(best_result[1])
    return features_to_eliminate

def main(configuration=configurations, 
         csv_file='../data/annotations/Kiwi_train_only_labeled_Kiwi_NIR.csv', 
         image_count=162, 
         target_label='smart_ripeness', 
         stat_use=1, 
         x_axis=4, 
         y_axis=100, 
         wavelengths=252, 
         wavelength_factor=2,
         target_statistic='MAE'):
    return 
    

if __name__ == "__main__":
    main()
