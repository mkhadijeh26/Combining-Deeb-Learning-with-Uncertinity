import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the trained neural network
model = load_model(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Software codes\MATLAB & Python\NN - G + D\Outputs')

# Load the input data from an Excel file
input_df = pd.read_excel(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Software codes\MATLAB & Python\NN - G + D\Data_for_NN.xlsx')

# Extract the input values as a numpy array
input_data = input_df.iloc[:, :11].values

# Define a function that takes input data and returns model predictions
def model_predict(data):
    return model.predict(data)

# Initialize a SHAP explainer object for the model and the input data
explainer = shap.DeepExplainer(model, input_data)

# Compute Shapley values for the input data
shap_values = explainer.shap_values(input_data)

# Create a list of the independent variables you want to test
independent_vars = ['Aging Temprature [°C]', 'Aging Time [h]', 'Aging Pressure [MPa]',
                 'Asphaltene (%) ', 'Napthene Aromatics (%)', 'Saturates (%)', 'Resins (%)',
                 'Penetration [dmm]', 'Softening Point [°C]', 'Test Temprature  [°C]',
                 'Test Frequency [rad/s]',]

# Calculate mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values[0]), axis=0)

# Create a DataFrame for plotting
df = pd.DataFrame(list(zip(independent_vars, mean_abs_shap_values)), columns=['Feature','Mean Abs SHAP Value'])
df = df.sort_values('Mean Abs SHAP Value', ascending=False)

# Plot feature importance based on mean absolute SHAP values
plt.figure(figsize=(10, 6), dpi=300)
plt.barh(df['Feature'], df['Mean Abs SHAP Value'], color='blue')
plt.xlabel('Mean Absolute SHAP Value',fontweight='bold', fontsize=14)
plt.title('Feature Importance',fontweight='bold', fontsize=14)
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.show()
