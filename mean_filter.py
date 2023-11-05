import pandas as pd
import numpy as np

# Load your dataset (assuming you have a DataFrame named 'data')
# Replace this with your actual data loading
data = pd.read_csv('E:\\GEETHA\\sit_stand_walk_test2.csv')

# Function to apply a moving average filter with window size 5
def moving_average(data, window_size):
    # Calculate the moving average of the real part of complex numbers
    real_part = data.apply(lambda x: x.apply(lambda z: z.real))
    moving_avg = real_part.rolling(window=window_size, min_periods=1).mean()
    return moving_avg.apply(lambda x: x.apply(lambda val: complex(val)))  # Restore complex numbers

# Apply moving average and data preprocessing to numeric columns (including complex numbers)
window_size = 5
numeric_columns = data.select_dtypes(include=[np.complex128]).columns  # Select complex numeric columns
data[numeric_columns] = data[numeric_columns].apply(lambda col: moving_average(col, window_size))
data = data.fillna(method='ffill')  # Forward fill NaN values with the nearest non-NaN value

# Save the preprocessed data to a new CSV file
output_file = 'E:\\GEETHA\\all_test2.csv'
data.to_csv(output_file, index=False)

print('Preprocessed data saved to', output_file)