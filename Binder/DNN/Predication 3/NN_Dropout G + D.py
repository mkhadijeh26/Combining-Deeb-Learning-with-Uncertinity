import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.regularizers import l2
import tensorflow as tf
import joblib

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Read the data from the Excel file
df = pd.read_excel(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\DNN\Predication 3\Data_for_NN.xlsx')

# Split the data into input features (X) and targets (y)
X = df.iloc[:, :-2].values
y = df.iloc[:, -2:].values


# Apply logarithmic transformation to targets
epsilon = 1e-6
y = np.log(y + epsilon)

# Scale the features and targets using MinMaxScaler
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed_value)

# Your directory path
dir_path = r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\DNN\Predication 3'

# Save the scalers in the specified directory
joblib.dump(X_scaler, f'{dir_path}\\X_scaler.pkl')
joblib.dump(y_scaler, f'{dir_path}\\y_scaler.pkl')

# Define the model with L2 regularization
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X.shape[1],))  
model.add(Dropout(0.2))  # Dropout 20% of the neurons
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))  # Add L2 regularization to this layer
model.add(Dropout(0.2))  # Dropout 20% of the neurons
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))  # Add L2 regularization to this layer
model.add(Dense(y.shape[1], activation='linear'))  # Output layer remains the same

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Predictions
y_train_pred_scaled = model.predict(X_train)
y_test_pred_scaled = model.predict(X_test)

# Inverse transform the predictions
y_train_pred = np.exp(y_scaler.inverse_transform(y_train_pred_scaled)) - epsilon
y_test_pred = np.exp(y_scaler.inverse_transform(y_test_pred_scaled)) - epsilon

# Inverse transform the original target values for comparison
y_train_original = np.exp(y_scaler.inverse_transform(y_train)) - epsilon
y_test_original = np.exp(y_scaler.inverse_transform(y_test)) - epsilon


# Visualize training and testing MSE vs. Epochs
plt.figure(figsize=(10, 8), dpi=300)
plt.plot(history.history['loss'], label='Training MSE', linewidth=2)
plt.plot(history.history['val_loss'], label='Testing MSE', linewidth=2)
plt.xlabel('Epochs', fontweight='bold', fontsize=20)
plt.ylabel('Mean Squared Error', fontweight='bold', fontsize=20)
plt.title('Training and Testing MSE vs. Epochs', fontweight='bold', fontsize=22)
plt.xticks(fontweight='bold', fontsize=18)
plt.yticks(fontweight='bold', fontsize=18)
plt.legend(fontsize=16, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.grid(False)
plt.tight_layout()
plt.savefig(f'{dir_path}\\MSEvsEpochs.png', dpi=300)
plt.show()

# Save the model
model.save(f'{dir_path}\\Outputs')

# Load the model
loaded_model = load_model(f'{dir_path}\\Outputs')

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
    plt.title(f'Training Set Predictions vs. True Values (Target {i+1}),  $\\mathbf{{R^2 = {r2:.2f}}}$', fontweight='bold', fontsize=18, va='top')
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(fontweight='bold', fontsize=18)
    plt.legend(fontsize=16, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_path}\\Training_Predictions_Target_{i+1}.png', dpi=300)
    plt.show()

# Repeat the plotting for the test set
for i in range(y_test_original.shape[1]):
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(y_test_original[:, i], y_test_pred[:, i], label='Testing', edgecolors='k', s=50, alpha=0.7)
    x_fit = np.linspace(min(y_test_original[:, i]), max(y_test_original[:, i]), 100)
    y_fit = x_fit
    plt.plot(x_fit, y_fit, 'r--', label='Ideal Fit', linewidth=2)
    plt.xlabel('True Values', fontweight='bold', fontsize=20)
    plt.ylabel('Predictions', fontweight='bold', fontsize=20)
    r2 = r2_score(y_test_original[:, i], y_test_pred[:, i])
    plt.title(f'Testing Set Predictions vs. True Values (Target {i+1}), $\\mathbf{{R^2 = {r2:.2f}}}$', fontweight='bold', fontsize=18, va='top')
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(fontweight='bold', fontsize=18)
    plt.legend(fontsize=16, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_path}\\Testing_Predictions_Target_{i+1}.png', dpi=300)
    plt.show()

# Calculate the mean squared error for the training set
mse_train = ((y_train_original - y_train_pred) ** 2).mean(axis=0)

# Calculate the mean squared error for the test set
mse_test = ((y_test_original - y_test_pred) ** 2).mean(axis=0)

# Print the MSE values
print(f"Mean Squared Error for Training Set: {mse_train}")
print(f"Mean Squared Error for Test Set: {mse_test}")
