import matplotlib.pyplot as plt
import numpy as np

filename = "position.txt"

# extract data from long list
n_agent = 12
data = np.fromfile(filename, count=-1, sep=',')
# print(data.shape)
T = int(np.floor(len(data) / 6))
data = data[T:T*2]
data = data.reshape([T//(n_agent*2), n_agent * 2])
# print(data)
# print(data[1, 0])
# print(data[:, 1][0])


colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'yellow', 'brown', 'gray', 'black']

plt.figure()

a = 0
# Update positions
for i in range(len(data[:, 0])):
    a += 1
    plt.clf()
    plt.scatter(data[i, 0], data[i, 1], s=100, marker='o')
    plt.scatter(data[i, 2], data[i, 3], s=100, marker='o')
    plt.scatter(data[i, 4], data[i, 5], s=100, marker='o')
    plt.scatter(data[i, 6], data[i, 7], s=100, marker='o')
    plt.scatter(data[i, 8], data[i, 9], s=100, marker='o')
    plt.scatter(data[i, 10], data[i, 11], s=100, marker='o')
    plt.scatter(data[i, 12], data[i, 13], s=100, marker='o')
    plt.scatter(data[i, 14], data[i, 15], s=100, marker='o')
    plt.scatter(data[i, 16], data[i, 17], s=100, marker='o')
    plt.scatter(data[i, 18], data[i, 19], s=100, marker='o')
    plt.scatter(data[i, 20], data[i, 21], s=100, marker='o')
    plt.scatter(data[i, 22], data[i, 23], s=100, marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('visualization')
    plt.pause(0.1)
    print(a)
plt.show()

