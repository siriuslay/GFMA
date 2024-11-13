import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'NCI1'     # choices=['REDDITBINARY', 'PROTEINS', 'MUTAG', 'PTC', 'COLLAB', 'IMDBBINARY', 'NCI1', 'REDDITMULTI5K']
split = 10
# 读取JSON文件
json_file = f'visual/results/continual_{dataset}_param_diffs_split_{split}.json'  # 你的JSON文件路径
with open(json_file, 'r') as f:
    all_diffs = json.load(f)

# 设置阈值
threshold = 0.001

# 计算每个模块中参数差异超过阈值的占比
module_names = list(all_diffs[0].keys())  # 获取模块名称（假设每次微调的模块一致）
num_tasks = len(all_diffs)  # 微调的轮次
num_modules = len(module_names)  # 模块数量

# 初始化存储占比的矩阵
change_ratios = np.zeros((num_modules, num_tasks))

# 遍历每次微调后的参数差异，计算超过阈值的占比
for task_idx, diffs in enumerate(all_diffs):
    for module_idx, module_name in enumerate(module_names):
        param_diffs = np.array(diffs[module_name])  # 转换为numpy数组
        num_params = param_diffs.size  # 参数数量
        num_above_threshold = np.sum(param_diffs > threshold)  # 超过阈值的参数数量
        change_ratios[module_idx, task_idx] = num_above_threshold / num_params  # 计算占比

# 生成热图
plt.figure(figsize=(12, 6))
sns.heatmap(change_ratios, cmap="Oranges", annot=False, cbar=True)
# sns.heatmap(change_ratios, cmap="YlGnBu", annot=False, cbar=True)
plt.xlabel('Task id')
plt.ylabel('Module id')
plt.title('Gradient Trend on NCI1')

# 设置横纵坐标刻度
plt.xticks(np.arange(0.5, num_tasks, 1), range(1, num_tasks+1), rotation=0)  # 横轴为微调轮次
plt.yticks(np.arange(0.5, num_modules, 1), module_names, rotation=0, fontsize=8)  # 纵轴为模块id

# 显示图像
plt.show()
plt.savefig(f'visual/results/{dataset}_split_{split}_param_changes_1e-3.pdf', format='pdf')
plt.close()

