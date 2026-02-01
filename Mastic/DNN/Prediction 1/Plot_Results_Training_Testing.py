import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import load_model
import joblib

# Define the directory path where the model and scalers are saved
dir_path = r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\Mastic\DNN\Prediction 5'

# Load the saved model and scalers
loaded_model = load_model(f'{dir_path}\\Outputs')
X_scaler = joblib.load(f'{dir_path}\\X_scaler.pkl')
y_scaler = joblib.load(f'{dir_path}\\y_scaler.pkl')

# Load the original data
df = pd.read_excel(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\Mastic\DNN\Prediction 5\Data_for_NN_ALLINPUTS.xlsx')

# Split the data into features (X) and targets (y)
X = df.iloc[:, :-2].values
y = df.iloc[:, -2:].values

# Apply the same logarithmic transformation to targets
epsilon = 1e-6
y_log = np.log(y + epsilon)

# Scale the features and targets using the loaded scalers
X_scaled = X_scaler.transform(X)
y_scaled = y_scaler.transform(y_log)

# Re-split the data into training and test sets (same random state as original)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Make predictions using the loaded model
y_train_pred_scaled = loaded_model.predict(X_train)
y_test_pred_scaled = loaded_model.predict(X_test)

# Inverse transform the predictions and true values
y_train_pred = np.exp(y_scaler.inverse_transform(y_train_pred_scaled)) - epsilon
y_test_pred = np.exp(y_scaler.inverse_transform(y_test_pred_scaled)) - epsilon
y_train_original = np.exp(y_scaler.inverse_transform(y_train)) - epsilon
y_test_original = np.exp(y_scaler.inverse_transform(y_test)) - epsilon

# Plotting predictions vs. true values for the training set
for i in range(y_train_original.shape[1]):
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(y_train_original[:, i], y_train_pred[:, i], label='Training', edgecolors='k', s=50, alpha=0.7)
    x_fit = np.linspace(min(y_train_original[:, i]), max(y_train_original[:, i]), 100)
    y_fit = x_fit
    plt.plot(x_fit, y_fit, 'r--', label='Ideal Fit', linewidth=2)
    plt.xlabel('True Values', fontweight='bold', fontsize=20)
    plt.ylabel('Predictions', fontweight='bold', fontsize=20)
    r2 = r2_score(y_train_original[:, i], y_train_pred[:, i])
    plt.title(f'Training Predictions vs. True Values (Target {i+1}), $\\mathbf{{R^2 = {r2:.2f}}}$', 
              fontweight='bold', fontsize=18, va='top')
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(fontweight='bold', fontsize=18)
    plt.legend(fontsize=16, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_path}\\Reloaded_Training_Predictions_Target_{i+1}.png', dpi=300)
    plt.show()

# Plotting predictions vs. true values for the test set
for i in range(y_test_original.shape[1]):
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(y_test_original[:, i], y_test_pred[:, i], label='Testing', edgecolors='k', s=50, alpha=0.7)
    x_fit = np.linspace(min(y_test_original[:, i]), max(y_test_original[:, i]), 100)
    y_fit = x_fit
    plt.plot(x_fit, y_fit, 'r--', label='Ideal Fit', linewidth=2)
    plt.xlabel('True Values', fontweight='bold', fontsize=20)
    plt.ylabel('Predictions', fontweight='bold', fontsize=20)
    r2 = r2_score(y_test_original[:, i], y_test_pred[:, i])
    plt.title(f'Testing Predictions vs. True Values (Target {i+1}), $\\mathbf{{R^2 = {r2:.2f}}}$', 
              fontweight='bold', fontsize=18, va='top')
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(fontweight='bold', fontsize=18)
    plt.legend(fontsize=16, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_path}\\Reloaded_Testing_Predictions_Target_{i+1}.png', dpi=300)
    plt.show()

# Export results to Excel
train_results = pd.DataFrame({
    'Train_True_Target1': y_train_original[:, 0],
    'Train_Pred_Target1': y_train_pred[:, 0],
    'Train_True_Target2': y_train_original[:, 1],
    'Train_Pred_Target2': y_train_pred[:, 1]
})

test_results = pd.DataFrame({
    'Test_True_Target1': y_test_original[:, 0],
    'Test_Pred_Target1': y_test_pred[:, 0],
    'Test_True_Target2': y_test_original[:, 1],
    'Test_Pred_Target2': y_test_pred[:, 1]
})

# Combine into a single Excel file with multiple sheets
with pd.ExcelWriter(f'{dir_path}\\Reloaded_Prediction_Results.xlsx') as writer:
    train_results.to_excel(writer, sheet_name='Training_Results', index=False)
    test_results.to_excel(writer, sheet_name='Testing_Results', index=False)

print(f"Results exported to {dir_path}\\Reloaded_Prediction_Results.xlsx")