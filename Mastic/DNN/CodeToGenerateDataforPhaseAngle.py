import pandas as pd
import numpy as np
import re

def assign_random_number(value):
    try:
        # Try to convert the value to float
        float_value = float(value)
        
        # Determine the order of magnitude
        if 1e3 <= float_value < 1e4:
            exponent = 3
            range_min, range_max = 85, 88
        if 1e4 <= float_value < 1e5:
            exponent = 4
            range_min, range_max = 80, 84
        if 1e5 <= float_value < 1e6:
            exponent = 5
            range_min, range_max = 70, 79
        elif 1e6 <= float_value < 1e7:
            exponent = 6
            range_min, range_max = 55, 69
        elif 1e7 <= float_value < 1e8:
            exponent = 7
            range_min, range_max = 35, 53
        elif 1e8 <= float_value < 1e9:
            exponent = 8
            range_min, range_max = 18, 33
        else:
            return np.nan  # Return NaN for values outside the specified ranges
        
        # Calculate the position within the range (0 to 1)
        position = 1 - ((float_value - 10**exponent) / (9 * 10**exponent))
        
        # Assign random number based on position
        return range_min + position * (range_max - range_min)
    
    except ValueError:
        # If conversion to float fails, return NaN
        return np.nan

# Load the Excel file
file_path = r"M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\Mastic\DNN\Prediction 1\Data.xlsx"
df = pd.read_excel(file_path)

# Check if the first column is in the correct order
first_column = df.iloc[:, 0]
if not all(first_column.sort_values().reset_index(drop=True) == first_column):
    print("Warning: The first column is not in ascending order.")

# Apply the function to assign random numbers
df['Assigned Number'] = df.iloc[:, 0].apply(assign_random_number)

# Save the updated DataFrame back to Excel
output_file_path = r"M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\Mastic\DNN\Prediction 1\Data_with_assigned_numbers.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Processed data saved to {output_file_path}")