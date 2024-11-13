import numpy as np
import matplotlib.pyplot as plt

# 设置菱形的顶点坐标
diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])  # 菱形的顶点

# 创建图形
plt.figure(figsize=(6, 6))
plt.plot(diamond[:, 0], diamond[:, 1], color='blue', linewidth=2)  # 绘制菱形边框

# 填充菱形内部
plt.fill(diamond[:, 0], diamond[:, 1], color='skyblue', alpha=0.5)

# 设置坐标轴比例相等
plt.axis('equal')

# 设置坐标轴范围
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

# 添加标题
plt.title("Diamond Shape")

# 显示图形
plt.grid()
plt.show()
plt.savefig(f'visual/results/diamond_0003.png', dpi=1200)
plt.close()