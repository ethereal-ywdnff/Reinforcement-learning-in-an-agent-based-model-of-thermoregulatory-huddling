import numpy as np
import pylab as pl

filename = "output.txt"

# extract data from long list
N = 12
data = np.fromfile(filename, count=-1, sep=',')
T = int(np.floor(len(data) / (N * 4 + 3)))
print(T)
data = data[0:(T * (N * 4 + 3))]
data = data.reshape([T, N * 4 + 3])
bt = 0
b_t = np.zeros(T)

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
    b_t[b] = bt / N
    bt = 0
sorted_bt = sorted(b_t)
print(b_t)
print(sorted_bt)

# example plots


# Fig1 = pl.figure()
# f1 = Fig1.add_subplot(311)
# f1.plot(Fsorted[:,1:],'k.')
# f1.set_ylabel('fitness')
# f2 = Fig1.add_subplot(312)
# f2.plot(Gsorted[:,1],'-')
# f2.set_ylabel('G_min')
# f3 = Fig1.add_subplot(313)
# f3.hist(Gsorted[:,1])
# f3.set_ylabel('histogram (G_min)')

# Fig2 = pl.figure()
# f1 = Fig2.add_subplot(211)
# f1.plot(Huddling)
# f1.set_ylabel('huddling')
# f2 = Fig2.add_subplot(212)
# f2.plot(PupFlow)
# f2.set_ylabel('pup flow')


temp_vec = np.linspace(10, 30, T)
Fig3 = pl.figure()
f1 = Fig3.add_subplot(111)
f1.plot(temp_vec, sorted_bt, '-')
f1.set_xlabel('body temperature')
f1.set_ylabel('body temperature')
pl.show()