import matplotlib.pyplot as plt
import numpy as np

filename = "position.txt"
# filename = "position_noLearning.txt"
# filename = "position_Learning.txt"

# extract data from long list
n_agent = 12
data = np.fromfile(filename, count=-1, sep=',')
print(data.shape)
T = int(np.floor(len(data)))
# data = data[0:T]
data = data.reshape([T//(n_agent*3), n_agent * 3])
# data = data[0:36036]
# data = data.reshape([36036//(n_agent*3), n_agent * 3])
# print(data.shape)
# print(data[1, 0])
# print(data[:, 1][0])


# colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'yellow', 'brown', 'gray', 'black']
cmap = plt.get_cmap('coolwarm')

plt.figure(figsize=(6, 6))


a = 0
# Update positions
for i in range(len(data[:, 0])):
    a += 1
    plt.clf()
    plt.xlim(-8, 10)
    plt.ylim(-11.5, 8)
    plt.scatter(data[i, 0], data[i, 1], s=700, color = cmap(data[i, 2] / 40), marker='o')
    plt.scatter(data[i, 3], data[i, 4], s=700, color = cmap(data[i, 5] / 40), marker='o')
    plt.scatter(data[i, 6], data[i, 7], s=700, color = cmap(data[i, 8] / 40), marker='o')
    plt.scatter(data[i, 9], data[i, 10], s=700, color = cmap(data[i, 11] / 40), marker='o')
    plt.scatter(data[i, 12], data[i, 13], s=700, color = cmap(data[i, 14] / 40), marker='o')
    plt.scatter(data[i, 15], data[i, 16], s=700, color = cmap(data[i, 17] / 40), marker='o')
    plt.scatter(data[i, 18], data[i, 19], s=700, color = cmap(data[i, 20] / 40), marker='o')
    plt.scatter(data[i, 21], data[i, 22], s=700, color = cmap(data[i, 23] / 40), marker='o')
    plt.scatter(data[i, 24], data[i, 25], s=700, color = cmap(data[i, 26] / 40), marker='o')
    plt.scatter(data[i, 27], data[i, 28], s=700, color = cmap(data[i, 29] / 40), marker='o')
    plt.scatter(data[i, 30], data[i, 31], s=700, color = cmap(data[i, 32] / 40), marker='o')
    plt.scatter(data[i, 33], data[i, 34], s=700, color = cmap(data[i, 35] / 40), marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('visualization')
    plt.pause(0.1)
    print(a)
plt.show()

