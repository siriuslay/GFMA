import numpy as np
import matplotlib.pyplot as plt
from math import pi
from matplotlib.lines import Line2D

# 设置类别标签
categories = ['MUTAG', 'PTC', 'REDDIT-B', 'NCI1']
N = len(categories)

# 不同模型的数据
values1 = [53.88, 48.67, 69.6, 60.5]
values2 = [51.64, 37.97, 32.81, 63.55]
values3 = [55.71, 77.19, 85.42, 66.97]
values4 = [53.43, 37.5, 82.34, 64.0]
values5 = [52.67, 35.31, 86.36, 65.28]
values6 = [55.6, 50, 80.36, 66.23]

# 重新排列数据以适应雷达图
values1 += values1[:1]
values2 += values2[:1]
values3 += values3[:1]
values4 += values4[:1]
values5 += values5[:1]
values6 += values6[:1]

# 设置角度
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# 创建图形并调整大小
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 设置坐标系背景颜色
ax.set_facecolor('#f0f0f0')

# 绘制不同模型
# ax.fill(angles, values1, color='#ADBCD3', alpha=0.15, label='w/o MoE')
# ax.plot(angles, values1, color='#A6BCD3', linewidth=2, marker='o')

# ax.fill(angles, values2, color='#F7C796', alpha=0.15, label=r'w/o $\mathcal{L}_{gen}$')
# ax.plot(angles, values2, color='#F7C796', linewidth=2, marker='o')

# ax.fill(angles, values3, color='#F0ACAD', alpha=0.15, label='w/o Graph-free')
# ax.plot(angles, values3, color='#F0ACAD', linewidth=2, marker='o')

# ax.fill(angles, values4, color='#BADAD7', alpha=0.15, label='w/o Graph-free & mask')
# ax.plot(angles, values4, color='#BADAD7', linewidth=2, marker='o')

# ax.fill(angles, values5, color='#ABD0A5', alpha=0.15, label='w/o mask')
# ax.plot(angles, values5, color='#ABD0A5', linewidth=2, marker='o')

# ax.fill(angles, values6, color='#F6E4A6', alpha=0.15, label='GFMA')
# ax.plot(angles, values6, color='#F6E4A6', linewidth=2, marker='o')

ax.fill(angles, values1, color='#DEAF6F', alpha=0.15, label='w/o MoE')
ax.plot(angles, values1, color='#DEAF6F', linewidth=2, marker='o')

ax.fill(angles, values2, color='#A48BD3', alpha=0.15, label=r'w/o $\mathcal{L}_{gen}$')
ax.plot(angles, values2, color='#A48BD3', linewidth=2, marker='o')

ax.fill(angles, values3, color='#ADDEAD', alpha=0.15, label='w/o Graph-free')
ax.plot(angles, values3, color='#ADDEAD', linewidth=2, marker='o')

ax.fill(angles, values4, color='#88ACD3', alpha=0.15, label='w/o Graph-free & mask')
ax.plot(angles, values4, color='#88ACD3', linewidth=2, marker='o')

ax.fill(angles, values5, color='#D7B0AF', alpha=0.15, label='w/o mask')
ax.plot(angles, values5, color='#D7B0AF', linewidth=2, marker='o')

ax.fill(angles, values6, color='#DE777C', alpha=0.15, label='GFMA')
ax.plot(angles, values6, color='#DE777C', linewidth=2, marker='o')


# 添加类别标签，调整位置以避免覆盖
plt.xticks(angles[:-1], categories)
# plt.text(angles[0], values1[0] + 5, categories[0], horizontalalignment='center')
# plt.text(angles[2], values2[2] + 5, categories[2], horizontalalignment='center')
# plt.xticks(angles[:-1], [])

# 添加图例，设置透明度和图例位置
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

# 创建图例，使用线条
legend_lines = [
    Line2D([0], [0], color='#DEAF6F', lw=2, label='w/o MoE'),
    Line2D([0], [0], color='#A48BD3', lw=2, label=r'w/o $\mathcal{L}_gen$'),
    Line2D([0], [0], color='#ADDEAD', lw=2, label='w/o Graph-free'),
    Line2D([0], [0], color='#88ACD3', lw=2, label='w/o Graph-free & mask'),
    Line2D([0], [0], color='#D7B0AF', lw=2, label='w/o mask'),
    Line2D([0], [0], color='#DE777C', lw=2, label='GFMA'),
]

# 添加图例
plt.legend(handles=legend_lines, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

# 设置标题
# plt.title("(a) New York", size=14, y=1.1)

# 设置图形比例
ax.set_aspect('equal', 'box')

# 设置虚线刻度线，保留三条
ax.yaxis.grid(True, color='gray', alpha=0.5, linestyle='--', linewidth=1)
ax.set_yticks(np.linspace(0, 100, 4)[1:-1])  # 只保留中间两条刻度线

# 显示图形
plt.show()

plt.savefig(f'visual/results/ablation-ladar.pdf', format='pdf')
plt.close()
