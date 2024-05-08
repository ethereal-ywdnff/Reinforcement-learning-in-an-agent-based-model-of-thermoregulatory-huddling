"""
This is to plot graphs of huddling and association between agents for filial huddling.

python graph.py
"""

import numpy as np
import matplotlib.pyplot as plt

filename = "output.txt"

N = 12
data = np.fromfile(filename, count=-1, sep=',')
T = int(np.floor(len(data) / (N * 1 + 2)))
data = data[0:(T * (N * 1 + 2))]
data = data.reshape([T, N * 1 + 2])
bt = 0
body_temp = np.zeros(T)

data2 = np.fromfile("association.txt", count=-1, sep=',')

data2 = data2.reshape(len(data2)//(N*N), N*N)
print(data2.shape)
association = data2[:12000, 1:12]


ambient_temp = data[:, -1]
huddling = data[:,-2]
n = 10
huddling = huddling.reshape(n, len(huddling)//n)
huddling = huddling.mean(axis=0)

data3 = np.fromfile("learning.txt",count=-1,sep=',')
T1 = int(np.floor(len(data3) / (N * 1 + 2)))
data3 = data3[0:(T1 * (N * 1 + 2))]
data3 = data3.reshape([T1, N * 1 + 2])
huddling_learning = data3[:,-2]
huddling_learning = huddling_learning.reshape(n, len(huddling_learning)//n)
huddling_learning = huddling_learning.mean(axis=0)

data = data[:, 0:-2].reshape([T, N, 1])
B = data[:, :, 0]
Bsorted = np.zeros([T, 12])

for t in range(T):
    i = np.argsort(B[t, :])
    Bsorted[t] = B[t, i]

for b in range(T):
    for c in range(N):
        bt += np.sum(Bsorted[b, c])
    body_temp[b] = bt / N
    bt = 0

# sort the lists
ambient_temp = sorted(ambient_temp)
body_temp = sorted(body_temp)


Fig = plt.figure(figsize=(7, 7))


f1 = Fig.add_subplot(211)
f1.set_ylim(-0.5, 1)
f1.plot(association)
f1.set_xlabel('Iteration')
f1.set_ylabel('Association')

f2 = Fig.add_subplot(212)
f2.set_title("Huddling")
days = np.linspace(1, 60, 12)
days_l = np.arange(60)
f2.plot(days, huddling, label="no learning")
f2.plot(days, huddling_learning, label="learning")
f2.set_xlabel('Days')
f2.set_ylabel('Huddle Size')
f2.legend()

plt.tight_layout()
plt.show()