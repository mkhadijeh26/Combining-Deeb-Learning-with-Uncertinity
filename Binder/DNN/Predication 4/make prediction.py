import pandas as pd
import numpy as np
from keras.models import load_model
import joblib

# Define the path to your directory
dir_path = r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\DNN\Predication 4'

# Load the trained model
model = load_model(f'{dir_path}\\Outputs')

# Load the scalers
X_scaler = joblib.load(f'{dir_path}\\X_scaler.pkl')
y_scaler = joblib.load(f'{dir_path}\\y_scaler.pkl')


# Read the input data from the Excel file, selecting only the first 11 columns
input_data = pd.read_excel(f'{dir_path}\\Data_for_NN.xlsx', usecols=range(11))

# Prepare the input data
X_new = input_data.values

# Scale the input data
X_new_scaled = X_scaler.transform(X_new)

# Make predictions
y_pred_scaled = model.predict(X_new_scaled)

# Inverse transform the predictions
epsilon = 1e-6
y_pred = np.exp(y_scaler.inverse_transform(y_pred_scaled)) - epsilon

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame(y_pred, columns=['Target_1', 'Target_2'])

# Concatenate input data with predictions
result_df = pd.concat([input_data, predictions_df], axis=1)

# Save the predictions to a new Excel file
result_df.to_excel(f'{dir_path}\\Predictions_output.xlsx', index=False)

print("Predictions have been saved to 'Predictions_output.xlsx'")