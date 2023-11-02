import numpy as np
import matplotlib.pyplot as plt


a = 2

if a:
    print(a)
else:
    print("not a")


a, b, c, d = 0.0, 0.0, 0.0, 0.0

a = 1.2
c = 1.3
print(a)
print(c)



my_array = np.array([1, 2, 3, 4, 5])

for i in range(len(my_array)):
    print(my_array[i])



# 创建一个范围在10到40之间的示例数据
data = np.random.randint(10, 41, 100)

# 选择一个颜色映射，例如 'viridis'，你可以根据需要选择不同的颜色映射
cmap = plt.get_cmap('viridis')

# 根据数据值生成颜色
colors = [cmap(value / 40) for value in data]

# 绘制颜色标记
plt.scatter(1, 2, c=cmap(20 / 40), marker='o', s=100)

# 显示图形
plt.show()