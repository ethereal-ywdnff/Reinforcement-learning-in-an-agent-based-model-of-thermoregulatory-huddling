import matplotlib.pyplot as plt
import numpy as np

filename = "x_y.txt"

# extract data from long list
N = 12
data = np.fromfile(filename, count=-1, sep=',')
# print(data.shape)
T = int(np.floor(len(data) / 6))
data = data[0:T]
data = data.reshape([T//(N*2), N * 2])
print(data)
print(data[1, 0])
print(data[:, 1][0])
# 初始化12个代理的初始x和y坐标
num_agents = 12
x_coordinates = data[:, 0]
y_coordinates = data[:, 1]  # 随机生成y坐标

# 为每个代理指定颜色
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'yellow', 'brown', 'gray', 'black']

# 创建一个Matplotlib图形
plt.figure()

# 绘制代理的小圆圈，指定颜色
# plt.scatter(x_coordinates, y_coordinates, s=100, c=colors, marker='o')

# 设置图形属性
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('visualization')

# 更新代理的位置
for i in range(len(data[:, 0])):
    x = x_coordinates[i]
    y = y_coordinates[i]
    plt.clf()
    plt.scatter(data[i, 0], data[i, 1], s=100, marker='o')
    plt.scatter(data[i, 2], data[i, 3], s=100, marker='o')
    plt.scatter(data[i, 4], data[i, 5], s=100, marker='o')
    plt.scatter(data[i, 6], data[i, 7], s=100, marker='o')
    plt.scatter(data[i, 8], data[i, 9], s=100, marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('visualization')
    plt.pause(0.1)

plt.show()

