import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



npz = np.load("/devdata/ClothesDataset_resampled_15000"+"/"+"wb_dress_sleeveless_147.npz")

pcds=npz['pcds']
selected_coords=npz['selected_coords']

print(len(pcds))

# 假设点云数据存储在名为"point_cloud_data.npy"的NumPy数组中
point_cloud_data = pcds

# 分离点的x、y、z坐标
x = point_cloud_data[:, 0]
y = point_cloud_data[:, 1]
z = point_cloud_data[:, 2]

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制点云
ax.scatter(x, y, z, c='b', marker='.')

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()
