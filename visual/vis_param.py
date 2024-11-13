import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = {
    "gate_p": [0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 100, 100, 100, 100, 100],
    "ce_p": [0.01, 0.1, 1, 10, 100, 0.01, 0.1, 1, 10, 100, 0.01, 0.1, 1, 10, 100, 0.01, 0.1, 1, 10, 100, 0.01, 0.1, 1, 10, 100],
    # "results": [66.29, 66.41, 66.34, 66.12, 66.09, 66.29, 66.39, 66.34, 66.12, 66.35, 
    #             66.28, 66.36, 66.32, 66.1, 66.35, 66.32, 66.36, 66.35, 66.13, 66.35, 
    #             66.19, 66.34, 66.28, 66.36, 66.35]
    "results": [80.06, 80.04, 80.16, 79.52, 79.34, 80.06, 80.06, 80.01, 79.52, 79.34, 80.11, 80.05, 79.52, 79.52, 79.34, 71.11, 71.11, 71.11, 70.24, 68.2, 34.79, 34.79, 34.79, 34.76, 32.6]

}

df = pd.DataFrame(data)

# Map values for custom tick labels
mapping = {0.01: 1, 0.1: 2, 1: 3, 10: 4, 100: 5}
df['gate_p_mapped'] = df['gate_p'].map(mapping)
df['ce_p_mapped'] = df['ce_p'].map(mapping)

# Prepare data for heatmap by creating pivot table
heatmap_data = df.pivot(index="ce_p_mapped", columns="gate_p_mapped", values="results")

# Calculate the mean values for each row and column
mean_x = df.groupby("gate_p_mapped")["results"].mean()
mean_y = df.groupby("ce_p_mapped")["results"].mean()

# Plot with adjusted layout to ensure alignment between line plots and heatmap
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)

# Main heatmap area
ax_heatmap = fig.add_subplot(gs[1:4, 0:3])
sns.heatmap(heatmap_data, cmap='viridis', cbar=True, ax=ax_heatmap, annot=True, fmt=".2f", cbar_kws={'location': 'right', 'label': 'Accuracy(%)'})

# Set custom ticks for the heatmap
ax_heatmap.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
ax_heatmap.set_xticklabels([0.01, 0.1, 1, 10, 100])
ax_heatmap.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
ax_heatmap.set_yticklabels([0.01, 0.1, 1, 10, 100])
ax_heatmap.set_xlabel(r"$\lambda_{gate}$")
ax_heatmap.set_ylabel(r"$\lambda_{mask}$")

# # Top line plot for column mean values, aligned with heatmap columns
# ax_line_x = fig.add_subplot(gs[0, 0:3])
# ax_line_x.plot([0.5, 1.5, 2.5, 3.5, 4.5], mean_x.values, color='olive', marker='o', markersize=6, linewidth=1.5)
# ax_line_x.set_xlim(0, 5)  # Align x-axis with heatmap grid
# ax_line_x.set_xticks([])  # Remove x-tick labels for clarity
# ax_line_x.set_ylabel("Mean Accuracy")
# ax_line_x.spines['bottom'].set_visible(False)  # Hide the bottom spine for better alignment

# # Right line plot for row mean values, aligned with heatmap rows
# ax_line_y = fig.add_subplot(gs[1:4, 3])
# ax_line_y.plot(mean_y.values, [0.5, 1.5, 2.5, 3.5, 4.5], color='olive', marker='o', markersize=6, linewidth=1.5)
# ax_line_y.set_ylim(0, 5)  # Align y-axis with heatmap grid
# ax_line_y.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
# ax_line_y.set_yticklabels([])  # Remove y-tick labels for clarity
# ax_line_y.set_xlabel("Mean Accuracy")
# ax_line_y.spines['left'].set_visible(False)  # Hide the left spine for better alignment

plt.show()
plt.savefig(f'visual/results/param_sensitive_redditb.pdf', format='pdf')   # redditb  nci1
plt.close()