import pandas as pd
import matplotlib.pyplot as plt

# Load real and synthetic data from Excel files
real_data = pd.read_excel("Data_for_NN.xlsx")
# Create a figure with subplots for each feature
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))

# Compare the distributions of each feature in the real and synthetic data using histograms
for i, feature in enumerate(real_data.columns):
    row = i // 4
    col = i % 4
    axs[row, col].hist(real_data[feature], alpha=0.5, label='Real Data')
    axs[row, col].set_title(feature)
    axs[row, col].legend()

# Adjust spacing between subplots and display the figure
fig.tight_layout()
plt.show()
