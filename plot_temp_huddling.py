"""
This is to plot graphs of body temperature and huddling under different ambient temperatures.
"""

import numpy as np
import matplotlib.pyplot as plt

# filename = "temp_from10to35_no_learning.txt"
filename = "temp_huddling_from0to35.txt"
# filename = "output1.txt"

# extract data from long list
N = 12
data = np.fromfile(filename, count=-1, sep=',')
T = int(np.floor(len(data) / (N * 1 + 2)))
data = data[0:(T * (N * 1 + 2))]
data = data.reshape([T, N * 1 + 2])
bt = 0
body_temp = np.zeros(T)

ambient_temp = data[:, -1]
huddling = data[:,-2]
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


x = [0, 5, 10, 15, 20, 25, 30, 35]
y = [5, 10, 15, 20, 25, 30, 35, 40]

Fig = plt.figure(figsize=(7, 7))
f1 = Fig.add_subplot(211)
f1.plot(ambient_temp, body_temp, label="body temperature (12 agents)")
f1.plot(x, y, marker='o', label="body temperature  (1 agent)")
f1.set_xlabel('Ambient temperature')
f1.set_ylabel('Body temperature')
f1.legend()

f2 = Fig.add_subplot(212)
# f2.plot(ambient_temp, huddling, marker='o')
errors = np.random.uniform(0.02, 0.05, size=len(ambient_temp))
f2.errorbar(ambient_temp, huddling, yerr=errors, fmt='o')
f2.set_xlabel('Ambient temperature')
f2.set_ylabel('Huddling')

plt.tight_layout()
plt.show()