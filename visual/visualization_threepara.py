import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成一些示例数据
np.random.seed(42)
num_points = 30
omega_d = np.random.uniform(0, 1.2, num_points)
omega_c = np.random.uniform(0, 1.2, num_points)
omega_ps = np.random.uniform(0, 1.2, num_points)

# 颜色数据 (以 z 轴值或其他度量来表示)
colors = np.random.rand(num_points)

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 散点图
sc = ax.scatter(omega_d, omega_c, omega_ps, c=colors, cmap='RdYlGn', s=100)

# 添加色条
plt.colorbar(sc)

# 设置轴标签
ax.set_xlabel(r'$\omega_d$')
ax.set_ylabel(r'$\omega_c$')
ax.set_zlabel(r'$\omega_{ps}$')

# 设置显示的视角
ax.view_init(30, 120)

# 添加网格平面
xx, yy = np.meshgrid(np.linspace(0, 1.2, 10), np.linspace(0, 1.2, 10))
xx, yy = np.meshgrid(np.linspace(0, 1.2, 20), np.linspace(0, 1.2, 20))
zz = 1 - xx - yy  # z = 1 - x - y
zz[zz < 0] = np.nan  # 去掉 z < 0 的部分，避免图中出现下方平面
ax.plot_surface(xx, yy, zz, alpha=0.3)

# 标题
ax.set_title("(a) New York")

plt.show()
plt.savefig(f'visual/results/three_para_0001.png', dpi=1600)
plt.close()