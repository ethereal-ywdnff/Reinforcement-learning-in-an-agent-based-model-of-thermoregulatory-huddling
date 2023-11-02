import numpy as np
import matplotlib.pyplot as plt

filename = "output.txt"

# extract data from long list
N = 12
data = np.fromfile(filename, count=-1, sep=',')
T = int(np.floor(len(data) / (N * 4 + 3)))
data = data[0:(T * (N * 4 + 3))]
data = data.reshape([T, N * 4 + 3])
bt = 0
body_temp = np.zeros(T)

ambient_temp = data[:, -3]
Huddling = data[:, -2]
PupFlow = data[:, -1]

data = data[:, 0:-3].reshape([T, N, 4])

G = data[:, :, 0]
K = data[:, :, 1]
B = data[:, :, 2]
F = data[:, :, 3]

# sort agents by fitness (ascending)
Gsorted = np.zeros([T, 12])
Ksorted = np.zeros([T, 12])
Bsorted = np.zeros([T, 12])
Fsorted = np.zeros([T, 12])
for t in range(T):
    i = np.argsort(F[t, :])
    Gsorted[t] = G[t, i]
    Ksorted[t] = K[t, i]
    Bsorted[t] = B[t, i]
    Fsorted[t] = F[t, i]

for b in range(T):
    for c in range(N):
        bt += np.sum(Bsorted[b, c])
    body_temp[b] = bt / N
    bt = 0
# sorted_bt = sorted(b_t)
print(f"ambient_temp: {ambient_temp}")
print(f"body_temp: {body_temp}")
# print(sorted_bt)

temp_vec = np.linspace(10, 22, T)
Fig = plt.figure()
f1 = Fig.add_subplot(111)
f1.plot(ambient_temp, body_temp)
f1.set_xlabel('Ambient temperature')
f1.set_ylabel('Body temperature')
for i, (x, y) in enumerate(zip(ambient_temp, body_temp)):
    f1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 6), ha='center', va='center')
plt.show()