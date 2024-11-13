import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# import colorcet as cc

id = 4
domain = 'D'
# 示例数据
categories = ['Domain A', 'Domain B', 'Domain C', 'Domain D']
group1_1 = [95.74, 95.74, 59.57, 25.53]  # PTM1
group1_2 = [91.49, 93.62, 61.7, 19.15]  # PTM2
group1_3 = [74.47, 82.98, 70.21, 44.68]  # PTM3
group1_4 = [8.51, 6.38, 38.3, 80.85]  # PTM4

# mask PTM
group2_1 = [92.77, 94.04, 73.4, 70.64]  # mask PTM1
group2_std_1 = [2.37, 2.98, 4.69, 4.64]  # mask PTM1

group2_2 = [92.77, 95.96, 70.85, 72.77]  # mask PTM2
group2_std_2 = [1.70, 1.49, 3.81, 4.13]  # mask PTM2
#
group2_3 = [91.70, 94.26, 69.79, 74.47]  # mask PTM3
group2_std_3 = [2.60, 3.30, 3.90, 4.36]  # mask PTM3

group2_4 = [93.19, 95.11, 67.23, 74.04]  # mask PTM4
group2_std_4 = [1.28, 1.36, 4.38, 6.58]  # mask PTM4

# mask CONV PTM
group3_1 = [92.77, 92.77, 71.70, 73.19]  # mask CONV PTM1
group3_std_1 = [3.71, 4.58, 2.34, 9.33]  # mask CONV PTM1

group3_2 = [87.87, 90.0, 67.87, 71.28]  # mask CONV PTM2
group3_std_2 = [6.10, 4.04, 4.80, 5.65]  # mask CONV PTM2
#
group3_3 = [93.19, 94.26, 69.79, 75.96]  # mask CONV PTM3
group3_std_3 = [1.59, 4.04, 4.64, 5.47]  # mask CONV PTM3

group3_4 = [91.49, 94.47, 70.21, 75.74]  # mask CONV PTM4
group3_std_4 = [3.16, 2.55, 3.30, 3.83]  # mask CONV PTM4

# MASK CL PTM
group4_1 = [91.28, 95.53, 60.64, 32.13]  # mask CL PTM1
group4_std_1 = [3.86, 1.77, 2.73, 4.61]  # mask CL PTM1

group4_2 = [85.74, 89.36, 65.32, 68.30]  # mask CL PTM2
group4_std_2 = [2.14, 5.63, 3.02, 5.50]  # mask CL PTM2
#
group4_3 = [88.94, 86.17, 62.55, 58.30]  # mask CL PTM3
group4_std_3 = [4.44, 2.73, 2.89, 9.09]  # mask CL PTM3

group4_4 = [88.51, 83.83, 56.81, 61.06]  # mask CL PTM4
group4_std_4 = [4.38, 5.72, 2.14, 19.08]  # mask CL PTM4

# colors = ['#CED6E7', '#91B2D7', '#778BBE', '#807C93', '#575C73']  # blue
# colors = ['#76A0AD', '#597C8B', '#DABE84', '#D9BDC3', '#C4D0CC']  # green /red
# colors = ['#DEAF6F', '#A48BD3', '#ADDEAD', '#DE777C', '#D7B0AF', '#88ACD3']
colors = ['#ADBCD3', '#F7C796', '#F0ACAD', '#BADAD7', '#ABD0A5', '#F6E4A6']
# x轴上的位置
num_groups = 4  # 每个类别的柱子数量
x = np.arange(len(categories))  # 类别的数量
width = 0.2  # 每个柱的宽度

# 创建图形和子图
fig, ax = plt.subplots()

group1_list = [group1_1, group1_2, group1_3, group1_4]
group2_list = [group2_1, group2_2, group2_3, group2_4]
group2_std_list = [group2_std_1, group2_std_2, group2_std_3, group2_std_4]
group3_list = [group3_1, group3_2, group3_3, group3_4]
group3_std_list = [group3_std_1, group3_std_2, group3_std_3, group3_std_4]
group4_list = [group4_1, group4_2, group4_3, group4_4]
group4_std_list = [group4_std_1, group4_std_2, group4_std_3, group4_std_4]

group1 = group1_list[id-1]
group2 = group2_list[id-1]
group2_std = group2_std_list[id-1]
group3 = group3_list[id-1]
group3_std = group3_std_list[id-1]
group4 = group4_list[id-1]
group4_std = group4_std_list[id-1]

# 绘制两组数据的柱状图
rects1 = ax.bar(x + 0 * width, group1, width * 0.9, label=f'No Mask', color=colors[0], capsize=5)
rects2 = ax.bar(x + 1 * width, group2, width * 0.9, yerr=group2_std, label=f'Masked All', color=colors[3], capsize=5)
rects3 = ax.bar(x + 2 * width, group3, width * 0.9, yerr=group3_std, label=f'Masked Conv-Layer', color=colors[2], capsize=5)
rects4 = ax.bar(x + 3 * width, group4, width * 0.9, yerr=group4_std, label=f'Masked Classifier', color=colors[1], capsize=5)


# 添加标签和标题
ax.set_xlabel(f'Fine-tuning for PTM-{id}(pretrained on domain {domain})')
ax.set_ylabel('ACC after Fine-tuning with Domain X')
ax.set_title(f'Influence of Masked Parts')
ax.set_xticks(x + width * (num_groups - 1) / 2)
ax.set_xticklabels(categories)
ax.legend()

# 添加数据标签函数
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# def add_labels(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# 给每个柱子添加数值标签
# add_labels(rects1)
# add_labels(rects2)
# add_labels(rects3)
# add_labels(rects4)

# 显示图形
plt.tight_layout()
plt.show()
plt.savefig(f'visual/results/MUTAG-PTM-{id}.pdf', format='pdf')
plt.close()