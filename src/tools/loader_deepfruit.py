import os
import math
import pandas as pd
import numpy as np
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats
from preprocessing import calculate_stats_size, preprocess_image

def process_spectral_data(csv_file, image_count, stat_use=1, x_axis=10, y_axis=10, wavelengths=51, wavelength_factor=4, pca=None, scaler=None):
 
    length_of_statistics = 0
    print(f"Parameters used: image_count={image_count}, stat_use={stat_use}, x_axis={x_axis}, y_axis={y_axis}, wavelengths={wavelengths}, wavelength_factor={wavelength_factor}")

    # Load the CSV file
    records = csv_file 

    # Filter the records by fruit type and non-empty image_path and target_label

    if stat_use:
        # Calculate the size of the statistics array
        stats_size, length_of_statistics = calculate_stats_size(x_axis, y_axis, wavelengths, wavelength_factor)
        all_stats = np.empty((0, stats_size))
    else:
        all_stats = []


    # Process the records
    for index, record in records.iterrows():

        # Construct file paths
        file_path = record['files.data_file']
        file_path = 'data/raw/' + file_path
        # Suppress NotGeoreferencedWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            # Open the image and load the data using rasterio
            with rasterio.open(file_path) as src:
                data = src.read()  # Read all bands

        if stat_use:
            # Preprocess the image data
            stats = preprocess_image(data, x_axis, y_axis, wavelength_factor)
            # Stack stats vertically onto all_stats
            all_stats = np.vstack((all_stats, stats))
        else:
            all_stats.append(data.flatten())

    if not stat_use:
        all_stats = np.array(all_stats)

    # Normalize the statistics
    if scaler is None:
        scaler = StandardScaler()
    all_stats_normalized = scaler.fit_transform(all_stats)
    
    # Apply PCA
    if pca is None:
        n_components = 0.999
        pca = PCA(n_components=n_components)
        all_stats_pca = pca.fit_transform(all_stats_normalized)
    else:
        all_stats_pca = pca.transform(all_stats_normalized)


    print(f"Loaded {len(records)} images.")
    if stat_use:
        print(f"Applied {length_of_statistics} stats.")
    print(f"Dimensions of all_stats: {all_stats.shape}. This represents the statistics of the spectral data.")
    print()
    return all_stats_pca, length_of_statistics, pca, scaler


def load_spectral_data(csv_file, image_count, stat_use=1, x_axis=10, y_axis=10, wavelengths=51, wavelength_factor=4):
    data, length_of_statistics, pca, scaler = process_spectral_data(csv_file, image_count, stat_use, x_axis, y_axis, wavelengths, wavelength_factor)
    # Feedback
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Principal components shape:", pca.components_.shape)
    print(f"Shape of all_stats_pca: {data.shape}")
    return data, length_of_statistics, pca, scaler
