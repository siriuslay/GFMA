import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Generate sample data
np.random.seed(0)
param1 = np.array([
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5
])
param2 = np.array([
1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5
])
param3 = np.array([
1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
])
accuracy = np.array([
66.2914, 66.4103, 66.3813, 66.3817, 66.0906, 66.420, 66.3904, 66.3411, 66.104, 66.0906, 66.420, 66.3605, 66.3411, 66.1203, 66.106, 66.3806, 66.3605, 66.3609, 66.1305, 66.1203, 66.3603, 66.3505, 66.420, 66.3616, 66.3519,
66.2914, 66.4103, 66.3411, 66.1203, 66.0906, 66.2914, 66.3904, 66.3411, 66.1203, 66.3519, 66.2812, 66.3605, 66.321, 66.104, 66.3519, 66.3211, 66.3609, 66.3512, 66.1305, 66.3519, 66.193, 66.3405, 66.2823, 66.3616, 66.3519,
66.0711, 66.3937, 66.0906, 66.362, 66.106, 66.2941, 66.3937, 66.0906, 66.3519, 66.106, 66.2941, 66.3937, 66.104, 66.3519, 66.106, 66.234, 66.3933, 66.0906, 66.3519, 66.106, 66.311, 66.3407, 66.4116, 66.3519, 66.420,
65.3121, 58.18, 65.3121, 65.6627, 66.031, 65.3121, 58.1, 58.12, 65.6925, 66.031, 65.332, 65.332, 65.332, 65.7224, 66.0608, 65.392, 65.392, 65.392, 65.7819, 66.0705, 65.881, 65.881, 65.881, 66.3511, 66.4403,
65.3121, 65.3121, 65.3121, 65.3121, 65.5928, 65.3121, 65.3121, 65.3121, 65.3121, 65.6126, 65.332, 65.332, 65.332, 65.332, 65.6522, 65.3721, 65.3721, 65.3721, 65.3721, 65.7419, 65.909, 65.909, 65.909, 65.909, 66.3214
])

green_cmap = LinearSegmentedColormap.from_list("green_shades", ["lightgreen", "darkgreen"])


# # 设置刻度区间
bins_x = np.linspace(0, 1, 6)  # X轴刻度
bins_y = np.linspace(0, 1, 6)  # Y轴刻度
bins_z = np.linspace(0, 1, 6)  # Z轴刻度

# # 计算每个刻度截面上的平均Accuracy
mean_accuracy_x = []
mean_accuracy_y = []
mean_accuracy_z = []

# # X轴上各个截面
for i in range(len(bins_x) - 1):
    mask = (param1 >= bins_x[i]) & (param1 < bins_x[i + 1])
    mean_accuracy_x.append(accuracy[mask].mean() if mask.any() else np.nan)

# Y轴上各个截面
for i in range(len(bins_y) - 1):
    mask = (param2 >= bins_y[i]) & (param2 < bins_y[i + 1])
    mean_accuracy_y.append(accuracy[mask].mean() if mask.any() else np.nan)

# Z轴上各个截面
for i in range(len(bins_z) - 1):
    mask = (param3 >= bins_z[i]) & (param3 < bins_z[i + 1])
    mean_accuracy_z.append(accuracy[mask].mean() if mask.any() else np.nan)

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color mapped to accuracy
sc = ax.scatter(param1, param2, param3, c=accuracy, cmap='viridis', s=50, alpha=0.8, edgecolor='w', depthshade=True, vmin=65, vmax=67)


# Adding color bar to indicate accuracy
cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Accuracy')



# Labels and title
ax.set_xlabel(r'$\lambda_m$')
ax.set_ylabel(r'$\lambda_g$')
ax.set_zlabel(r'$\lambda_c$')
ax.set_title('Hyper-Parameter Analysis on NCI1')



ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['0.01', '0.1', '1', '10', '100'])

ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])

ax.set_zticks([1, 2, 3, 4, 5])
ax.set_zticklabels(['0.01', '0.1', '1', '10', '100'])

# # X轴
x_centers = (bins_x[:-1] + bins_x[1:]) / 2
ax.bar(x_centers, mean_accuracy_x, zs=param2.min() - 0.1, zdir='y', width=0.05, color='darkgreen', alpha=0.6)

# Y轴
y_centers = (bins_y[:-1] + bins_y[1:]) / 2
ax.bar(y_centers, mean_accuracy_y, zs=param1.max() + 0.1, zdir='x', width=0.05, color='darkgreen', alpha=0.6)

# Z轴
z_centers = (bins_z[:-1] + bins_z[1:]) / 2
ax.bar(z_centers, mean_accuracy_z, zs=param3.max() + 0.1, zdir='x', width=0.05, color='darkgreen', alpha=0.6)


ax.view_init(elev=20, azim=45)

plt.show()
plt.savefig('visual/results/param_analysis 2.svg', format='svg')
plt.savefig('visual/results/param_analysis 2', dpi=1600)
plt.close()