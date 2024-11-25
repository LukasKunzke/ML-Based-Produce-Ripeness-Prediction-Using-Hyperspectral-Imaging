import os
import json
import math
import numpy as np
import rasterio
from sklearn.preprocessing import StandardScaler
import scipy.stats

def calculate_statistics(two_d_slice):

    return np.array([
        np.min(two_d_slice),  # 0 Minimum value in the 2D slice
        np.percentile(two_d_slice, 3),
        np.percentile(two_d_slice, 5),# 1 3rd percentile of the 2D slice, 3% of the data fall below this value
        np.percentile(two_d_slice, 10),
        np.percentile(two_d_slice, 15),
        np.percentile(two_d_slice, 20),# 2 10th percentile of the 2D slice, 10% of the data fall below this value
        np.percentile(two_d_slice, 25),
        np.percentile(two_d_slice, 30),
        np.percentile(two_d_slice, 35),
        np.percentile(two_d_slice, 40),
        np.percentile(two_d_slice, 45),# 3 25th percentile of the 2D slice, also known as the first quartile
        np.median(two_d_slice),  # 4 Median (middle) value of the 2D slice
        np.mean(two_d_slice),  # 5 Mean (average) value of the 2D slice
        np.percentile(two_d_slice, 55),
        np.percentile(two_d_slice, 60),
        np.percentile(two_d_slice, 65),
        np.percentile(two_d_slice, 70),
        np.percentile(two_d_slice, 75),
        np.percentile(two_d_slice, 80),
        np.percentile(two_d_slice, 85),# 6 75th percentile of the 2D slice, also known as the third quartile
        np.percentile(two_d_slice, 90),
        np.percentile(two_d_slice, 95),# 7 90th percentile of the 2D slice, 90% of the data fall below this value
        np.percentile(two_d_slice, 97),  # 8 97th percentile of the 2D slice, 97% of the data fall below this value
        np.max(two_d_slice),  # 9 Maximum value in the 2D slice
        np.ptp(two_d_slice),  # 10 Range of values (maximum - minimum) in the 2D slice
        np.var(two_d_slice),  # 11 Variance of the 2D slice, a measure of the spread of the set of values
        np.std(two_d_slice),  # 12 Standard deviation of the 2D slice, a measure of the amount of variation or dispersion of the set of values
        np.sum(two_d_slice),  # 13 Sum of all values in the 2D slice
        np.mean(np.abs(two_d_slice - np.mean(two_d_slice))),  # 14 Mean absolute deviation of the 2D slice, a measure of dispersion
        scipy.stats.skew(two_d_slice.flatten()),  # 15 Skewness of the flattened 2D slice, a measure of the asymmetry of the probability distribution
        scipy.stats.kurtosis(two_d_slice.flatten()),  # 16 Kurtosis of the flattened 2D slice, a measure of the "tailedness" of the probability distribution
    ])
    
def calculate_stats_size(x_axis, y_axis, wavelengths, wavelength_factor):
  
    # Create a sample 2D slice
    sample_2d_slice = np.random.rand(x_axis, y_axis)
    
    # Calculate statistics for the sample 2D slice
    statistics = calculate_statistics(sample_2d_slice)
    
    # Get the length of the statistics array
    length_of_statistics = len(statistics)
    
    # Calculate the number of wavelength groups
    number_of_waves = math.floor(wavelengths / wavelength_factor) + (1 if wavelengths % wavelength_factor != 0 else 0)
    
    # Calculate the total size of all_stats
    stats_size = length_of_statistics * number_of_waves
    
    return stats_size, length_of_statistics
 
def preprocess_image(image, x_axis, y_axis, wavelength_factor):

    data = image

    # Calculate the middle indices along the first two dimensions
    middle_x = data.shape[1] // 2
    middle_y = data.shape[2] // 2

    # Calculate the start and end indices for the slice
    start_x = middle_x - x_axis // 2
    end_x = start_x + x_axis
    start_y = middle_y - y_axis // 2
    end_y = start_y + y_axis

    # Extract the slice
    slice = data[:, start_x:end_x, start_y:end_y]

    # Initialize average as an empty list
    average = []

    # Loop over the third dimension of slice in steps of wavelength_factor
    for i in range(0, slice.shape[0], wavelength_factor):
        # Determine the end index for the current slice
        end_index = min(i + wavelength_factor, slice.shape[0])
        # Average the slices from i to end_index
        avg_slice = np.mean(slice[i:end_index, :, :], axis=0)
        average.append(avg_slice)

    # Convert average to a numpy array and stack along the third dimension
    slice = np.stack(average, axis=0)

    # Initialize stats as an empty numpy array
    stats = np.array([])

    # Iterate over the third dimension of the slice array
    for i in range(slice.shape[0]):
        # Get the 2D array corresponding to the current slice along the third dimension
        two_d_slice = slice[i, :, :]

        # Calculate statistics and append them to the stats array
        stats = np.append(stats, calculate_statistics(two_d_slice))

    return stats