import numpy as np
import matplotlib.pyplot as plt
from math import pi

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

# 创建图形
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))



# 绘制第一个模型
ax.fill(angles, values1, color='#A6BCD3', alpha=0.15, label='w/o MoE')
ax.plot(angles, values1, color='#A6BCD3', linewidth=2, marker='o')

# 绘制第二个模型
ax.fill(angles, values2, color='#F7C796', alpha=0.15, label=r'w/o $\mathcal{L}_gen$')
ax.plot(angles, values2, color='#F7C796', linewidth=2, marker='o')

# 绘制第三个模型
ax.fill(angles, values3, color='#F0ACAD', alpha=0.15, label='w/o Graph-free')
ax.plot(angles, values3, color='#F0ACAD', linewidth=2, marker='o')

# 绘制第四个模型
ax.fill(angles, values4, color='#BADAD7', alpha=0.15, label='w/o Graph-free & mask')
ax.plot(angles, values4, color='#BADAD7', linewidth=2, marker='o')

ax.fill(angles, values5, color='#ABD0A5', alpha=0.15, label='w/o mask')
ax.plot(angles, values5, color='#ABD0A5', linewidth=2, marker='o')

ax.fill(angles, values6, color='#F6E4A6', alpha=0.15, label='GFMA')
ax.plot(angles, values6, color='#F6E4A6', linewidth=2, marker='o')

# 添加类别标签
plt.xticks(angles[:-1], categories)

# 添加图例，设置透明度和图例位置
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)

# 设置标题
# plt.title("(a) New York", size=14, y=1.1)

# 设置图形比例
ax.set_aspect('equal', 'box')

# 设置透明度更低的网格线，增强可见性
ax.yaxis.grid(True, color='gray', alpha=0.5)

# 显示图形
plt.show()

plt.savefig(f'visual/results/ladar_0003.png', dpi=1200)
plt.close()