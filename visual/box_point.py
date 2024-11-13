import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 初始化数据：假设有3个不同的方法，每个方法生成不同数量的样本
methods = ['Method1', 'Method2', 'Method3']

# 模拟每个方法得到的不同数量样本的特征，标签和特殊属性
np.random.seed(0)
features_list = [
    np.random.rand(150, 100),  # Method1有150个样本，100维特征
    np.random.rand(200, 80),  # Method2有200个样本，80维特征
    np.random.rand(180, 120)  # Method3有180个样本，120维特征
]

labels_list = [
    np.random.randint(0, 3, 150),  # Method1的标签
    np.random.randint(0, 4, 200),  # Method2的标签
    np.random.randint(0, 2, 180)  # Method3的标签
]

special_feature_list = [
    np.random.rand(150) * 10 - 5,  # Method1的特殊属性，范围[-5, 5]
    np.random.rand(200) * 8 - 4,  # Method2的特殊属性，范围[-4, 4]
    np.random.rand(180) * 12 - 6  # Method3的特殊属性，范围[-6, 6]
]

# 存储整合后的数据
all_methods = []
all_tsne = []
all_special_features = []
all_labels = []

# 处理每个方法的数据
for i, (features, labels, special_feature) in enumerate(zip(features_list, labels_list, special_feature_list)):
    # 1. 对特征进行t-SNE降维（降为1维，用作横坐标的一部分）
    tsne = TSNE(n_components=1, random_state=0)
    tsne_results = tsne.fit_transform(features).flatten()

    # 2. 对特殊属性进行归一化处理
    scaler = MinMaxScaler()
    special_feature_normalized = scaler.fit_transform(special_feature.reshape(-1, 1)).flatten()

    # 3. 合并数据
    all_methods.extend([methods[i]] * len(labels))  # 记录当前方法名
    all_tsne.extend(tsne_results)  # 记录t-SNE结果
    all_special_features.extend(special_feature_normalized)  # 记录归一化的特殊属性
    all_labels.extend(labels)  # 记录标签

# 创建最终的DataFrame
data = pd.DataFrame({
    'Method': all_methods,
    't-SNE': all_tsne,
    'SpecialFeature': all_special_features,
    'Label': all_labels
})

# 使用 Seaborn 生成柱状散点图，横轴是不同方法，纵轴是归一化后的特殊属性
plt.figure(figsize=(10, 6))

# 通过 hue 来区分标签，jitter 使得散点图中的点更均匀分布
sns.stripplot(x='Method', y='SpecialFeature', data=data, hue='Label', jitter=True, palette="Set1", size=4)

# 设置标题和标签
plt.title('Samples Visualization Across Different Methods')
plt.xlabel('Method')
plt.ylabel('Normalized Special Feature')

# 显示图例
plt.legend(title='Label')

# 显示图表
plt.show()
