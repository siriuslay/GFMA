import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
num_points = 15
omega_t = np.linspace(0, 1, num_points)  # x 轴数据
omega_pt = 1 - omega_t                   # y 轴数据，满足 y = 1 - x 的关系

# 颜色数据 (与背景颜色保持一致，使用相同的渐变)
colors = omega_pt  # 颜色与 omega_pt 对应，从下到上渐变

# 创建图形
fig, ax = plt.subplots()

# 创建渐变背景：左下三角区域
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.abs(Y)  # 渐变方向竖着，从下到上

# 只填充左下角三角形区域
Z[X + Y > 1] = np.nan  # 只保留 x + y <= 1 的区域

# 绘制背景渐变
ax.contourf(X, Y, Z, cmap='RdYlGn', alpha=0.3)

# 绘制散点图 (与背景颜色类似)
sc = ax.scatter(omega_t, omega_pt, c=colors, cmap='RdYlGn', s=200)

# 添加色条
plt.colorbar(sc)

# 设置轴标签
ax.set_xlabel(r'$\omega_t$')
ax.set_ylabel(r'$\omega_{pt}$')

# 设置标题
ax.set_title("(a) New York")

# 设置图形比例
ax.set_aspect('equal', 'box')

plt.show()

plt.savefig(f'visual/results/two_para_0002.png', dpi=1600)
plt.close()