import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the trained model
model = tf.keras.models.load_model(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\DNN\Predication 1\Outputs')

# Load the input data from Excel
input_data = pd.read_excel(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\DNN\Predication 1\Data_for_NN.xlsx', usecols=range(11)).values.tolist()

# Load the scalers
X_scaler = joblib.load(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\DNN\Predication 1\X_scaler.pkl')
y_scaler = joblib.load(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\DNN\Predication 1\y_scaler.pkl')

# Normalize the input data using the loaded X_scaler
normalized_input = X_scaler.transform(np.array(input_data))

# Make predictions using the normalized input
predictions = model.predict(normalized_input)

# Denormalize the predictions using the inverse_transform method of y_scaler
denormalized_predictions = y_scaler.inverse_transform(predictions)

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame(denormalized_predictions, columns=['G*', 'Phase Angle'])

# Load the original data with pandas
data_df = pd.read_excel(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\DNN\Predication 1\Data_for_NN.xlsx')

# Concatenate the original data with the predictions
output_df = pd.concat([data_df, predictions_df], axis=1)

# Write the output to the same Excel file
output_df.to_excel(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\DNN\Predication 1\Data_for_NN.xlsx', index=False)
