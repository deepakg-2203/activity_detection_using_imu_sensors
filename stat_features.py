import pandas as pd
import numpy as np

# Load your dataset (replace 'your_dataset.csv' with the actual path to your dataset)
data = pd.read_csv('E:\GEETHA\walking.csv')


# Set window size and overlap
window_size = 100
overlap = 0.5

# Calculate the step size
step = int(window_size * (1 - overlap))

# Initialize a DataFrame to store the results
results = pd.DataFrame()

# Define a function to compute autocorrelation for a single lag (k)
def autocorr(y, lag=1):
    y = np.array(y).copy()
    y_bar = np.mean(y)
    denominator = sum((y - y_bar) ** 2)
    numerator_p1 = y[lag:] - y_bar
    numerator_p2 = y[:-lag] - y_bar
    numerator = sum(numerator_p1 * numerator_p2)
    return (numerator / denominator)

# Iterate through each column in your dataset
for column in data.columns:
    # Create windows
    windows = [data[column][i:i + window_size] for i in range(0, len(data) - window_size + 1, step)]

    # Calculate mean, median, std, MAD, max, min, IQR, and range for each window
    window_means = [window.mean() for window in windows]
    window_medians = [window.median() for window in windows]
    window_stds = [window.std() for window in windows]

    # Define a function to calculate MAD (Median Absolute Deviation) manually
    def calculate_mad(window):
        median = window.median()
        mad = (window - median).abs().median()
        return mad

    window_mads = [calculate_mad(window) for window in windows]
    window_max = [window.max() for window in windows]
    window_min = [window.min() for window in windows]

    # Define a function to calculate IQR
    def calculate_iqr(window):
        return window.quantile(0.75) - window.quantile(0.25)

    window_iqrs = [calculate_iqr(window) for window in windows]

    # Calculate autocorrelations for each window
    window_autocorrelations = [autocorr(window) for window in windows]

    # Add the results to the DataFrame
    results[f'{column}-mean'] = window_means
    results[f'{column}-median'] = window_medians
    results[f'{column}-std'] = window_stds
    results[f'{column}-mad'] = window_mads
    results[f'{column}-max'] = window_max
    results[f'{column}-min'] = window_min
    results[f'{column}-iqr'] = window_iqrs
    results[f'{column}-autocorr'] = window_autocorrelations

# Save the results to a CSV file
results.to_csv('E:\GEETHA\walking_test.csv', index=False)
train=pd.read_csv('E:\GEETHA\walking_test.csv')
print(train.shape)
print(train.isna().values.sum())