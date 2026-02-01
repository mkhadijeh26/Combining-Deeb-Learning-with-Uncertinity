import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# Load data
df_raw = pd.read_excel(r'M:\citg\se\WR\Moisture Damage\Mahmoud PhD\Publication\TwoScales Paper\Asphalt Binder Neural Network _ HeatMap\Data_for_Pearson Coefficient.xlsx')

# Compute correlation matrix
corrmat = df_raw.corr()

# Set high-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# Create PDF to save all plots
pdf_filename = 'correlation_analysis_high_quality.pdf'

with PdfPages(pdf_filename) as pdf:

    # === Method 1: Enhanced Heatmap ===
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corrmat, dtype=bool))

    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf',
              '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    cmap = sns.blend_palette(colors, n_colors=256, as_cmap=True)

    ax = sns.heatmap(corrmat, mask=mask, annot=True, cmap=cmap,
                     vmin=-1, vmax=1, fmt='.3f', linewidths=1, square=True,
                     cbar_kws={"shrink": .8, "aspect": 30})

    plt.title('Enhanced Correlation Heatmap\n(Lower Triangle Only)',
              fontsize=20, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight', facecolor='white')
    plt.close()

    # === Method 2: Circular Correlation Plot ===
    plt.figure(figsize=(16, 16))
    corr_values = corrmat.values
    features = corrmat.columns.tolist()
    n_features = len(features)
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False)
    ax = plt.subplot(111, projection='polar')

    for i in range(n_features):
        for j in range(i+1, n_features):
            corr_val = corr_values[i, j]
            if abs(corr_val) > 0.3:
                color = plt.cm.Reds(abs(corr_val)) if corr_val > 0 else plt.cm.Blues(abs(corr_val))
                linewidth = abs(corr_val) * 5
                ax.plot([angles[i], angles[j]], [1, 1], color=color, linewidth=linewidth, alpha=0.7)

    for i, (angle, feature) in enumerate(zip(angles, features)):
        ax.text(angle, 1.1, feature, rotation=np.degrees(angle) - 90 if np.pi/2 < angle < 3*np.pi/2 else np.degrees(angle),
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.scatter(angle, 1, s=100, c='black', zorder=10)

    ax.set_ylim(0, 1.2)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rgrids([])
    ax.set_thetagrids([])
    ax.spines['polar'].set_visible(False)

    plt.title('Circular Correlation Network\n(|r| > 0.3)',
              fontsize=18, fontweight='bold', pad=30)
    red_line = plt.Line2D([0], [0], color='red', linewidth=3, label='Positive Correlation')
    blue_line = plt.Line2D([0], [0], color='blue', linewidth=3, label='Negative Correlation')
    plt.legend(handles=[red_line, blue_line], loc='upper right', bbox_to_anchor=(1.3, 1.1))

    pdf.savefig(bbox_inches='tight', facecolor='white')
    plt.close()

    # === Method 3: Correlation Strength Distribution ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    mask = np.triu(np.ones_like(corrmat), k=1).astype(bool)
    correlations = corrmat.where(mask).stack().reset_index()
    correlations.columns = ['Variable 1', 'Variable 2', 'Correlation']

    ax1.hist(correlations['Correlation'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(correlations['Correlation'].mean(), color='red', linestyle='--',
                label=f'Mean: {correlations["Correlation"].mean():.3f}')
    ax1.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Correlation Coefficients', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.boxplot(correlations['Correlation'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax2.set_title('Box Plot of Correlations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Correlation Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight', facecolor='white')
    plt.close()

    # === Method 4: Top Correlations Bar Plot ===
    plt.figure(figsize=(14, 10))
    correlations_abs = correlations.copy()
    correlations_abs['Abs_Correlation'] = abs(correlations_abs['Correlation'])
    correlations_sorted = correlations_abs.sort_values('Abs_Correlation', ascending=False)
    top_correlations = correlations_sorted.head(20)
    labels = [f"{row['Variable 1']} - {row['Variable 2']}" for _, row in top_correlations.iterrows()]
    colors = ['red' if x > 0 else 'blue' for x in top_correlations['Correlation']]
    bars = plt.barh(range(len(top_correlations)), top_correlations['Correlation'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_correlations)), labels, fontsize=10)
    plt.xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    plt.title('Top 20 Variable Correlations', fontsize=16, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='x')

    for i, (bar, val) in enumerate(zip(bars, top_correlations['Correlation'])):
        plt.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}',
                 va='center', ha='left' if val > 0 else 'right', fontweight='bold')

    plt.tight_layout()
    pdf.savefig(bbox_inches='tight', facecolor='white')
    plt.close()

    # === Method 5: Hierarchical Clustering Heatmap (with manual annotations) ===
    plt.figure(figsize=(16, 12))
    distance_matrix = 1 - abs(corrmat)
    condensed_distances = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distances, method='ward')

    g = sns.clustermap(
        corrmat,
        method='ward',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,  # disable built-in annot
        linewidths=0.5,
        figsize=(16, 12),
        cbar_kws={"shrink": .8}
    )

    # Annotate manually after reordering
    reordered_rows = g.dendrogram_row.reordered_ind
    reordered_cols = g.dendrogram_col.reordered_ind
    corr_values_reordered = corrmat.values[np.ix_(reordered_rows, reordered_cols)]

    for i in range(len(reordered_rows)):
        for j in range(len(reordered_cols)):
            val = corr_values_reordered[i, j]
            g.ax_heatmap.text(j + 0.5, i + 0.5, f'{val:.2f}',
                  ha='center', va='center',
                  color='black', fontsize=15, fontweight='bold')

    g.fig.suptitle('Hierarchically Clustered Correlation Heatmap',
                   fontsize=18, fontweight='bold', y=1.02)

    pdf.savefig(g.fig, bbox_inches='tight', facecolor='white')
    plt.close()

print(f"High-quality correlation visualizations saved to: {pdf_filename}")
