import matplotlib.pyplot as plt
import numpy as np


# syn_ratio = [2, 3, 5, 7, 9, 11, 15]
# average_auc = [80.36, 82.6, 82.39, 82.0, 80.9, 78.3, 70.26]
# error = [1.64, 2.31, 0.46, 1.2, 1.33, 1.584, 1.92]  # 误差数据

# syn_ratio = [2, 3, 5, 7, 9, 11, 15]
# average_auc = [66.23, 30.9, 72.89, 74.45, 76.01, 47.67, 78.52]
# error = [0.15, 0.0, 0.11, 0.09, 1.418, 1.21, 1.332]  # 误差数据

# syn_ratio = [0.05, 0.25, 0.5, 0.75, 1.0]
# average_auc = [52.7, 82.0, 78.25, 81.3, 80.1]
# error = [1.33, 0.0, 1.586, 1.21, 1.57]  # 误差数据

syn_ratio = [0.05, 0.25, 0.5, 0.75, 1.0]
average_auc = [59.42, 76.01, 75.72, 75.4, 76.06]
error = [1.57, 0.46, 0.5, 0.9, 0.58]  # 误差数据

# NCI1	±	±	±	±

# 创建图形
fig, ax = plt.subplots(figsize=(7, 6))
#ADDEAD
#DE777C
# 绘制折线图和误差带
ax.plot(syn_ratio, average_auc, color='#DE777C', linewidth=2, marker='o')
ax.fill_between(syn_ratio, np.array(average_auc) - np.array(error), np.array(average_auc) + np.array(error), 
                color='#DE777C', alpha=0.2)

# 设置坐标轴标签
ax.set_xlabel('Syn_Ratio')
ax.set_ylabel('Average ACC on NCI1')

# 设置坐标轴范围
ax.set_ylim(40, 100)

# 设置网格线
ax.grid(True, linestyle='--', linewidth=0.5)

# 显示图形
plt.show()
plt.savefig(f'visual/results/syn_ratio-nci1.pdf', format='pdf')
plt.close()