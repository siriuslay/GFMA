import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义函数用于生成球面坐标
def sphere_points(num_points):
    """生成球面上均匀分布的点"""
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

# 创建绘图窗口和3D轴
fig = plt.figure(figsize=(10, 5))

# 绘制左图
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title(r'Training with $\mathcal{L}_{ce}$')

# 绘制球面网格
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3)

# 生成随机球面点并绘制
num_points = 1000
x, y, z = sphere_points(num_points)
ax1.scatter(x, y, z, c=np.random.rand(num_points), cmap='rainbow', s=5)

# 设置轴
ax1.set_box_aspect([1, 1, 1])

# 绘制右图
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title(r'Training with $\mathcal{L}_{ce} + \mathcal{L}^{p}_{con}$')

# 绘制球面网格
ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3)

# 生成新的球面点并绘制
x, y, z = sphere_points(num_points)
ax2.scatter(x, y, z, c=np.random.rand(num_points), cmap='rainbow', s=5)

# 设置轴
ax2.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()