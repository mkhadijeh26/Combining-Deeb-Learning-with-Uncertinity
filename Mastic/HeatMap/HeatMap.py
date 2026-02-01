import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df_raw = pd.read_excel(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\Mastic\HeatMap\Data_for_NN_ALLINPUTS.xlsx') 

# Calculate correlations
corrmat = df_raw.corr()
top_corr_features = corrmat.index

# Set up the matplotlib figure
plt.figure(figsize=(16, 14))

# Create the heatmap using seaborn
sns.set(font_scale=1.2)
heatmap = sns.heatmap(df_raw[top_corr_features].corr(), 
                      annot=True, 
                      cmap='RdYlGn', 
                      vmin=-1, 
                      vmax=1, 
                      fmt='.2f',
                      linewidths=0.5,
                      cbar_kws={"shrink": .8})

# Improve the appearance
plt.title('Correlation Heatmap', fontsize=24, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(rotation=0, fontsize=15)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show the plot
plt.show()
