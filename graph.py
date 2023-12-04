import numpy as np
import matplotlib.pyplot as plt

filename = "output.txt"
# filename = "output_template.txt"

# extract data from long list
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
association = data2[:, 12:24]
# print(association[:, :1].shape)
# print(association)

F = "learning.txt"
M = 3
K = 1
S = 5000
data3 = np.fromfile(F,count=-1,sep=',')
# print(data.shape)
N = int(data3[0])
# dt = data[1]
# print(dt)
data3 = data3[1:]
T1 = int(np.floor(len(data3)/(N*M+K)))
data3 = data3[0:(T1*(N*M+K))]
data3 = data3.reshape([T1,N*M+K])
time = data3[:,-1]
data3 = data3[:,:N*M].reshape([T1,N,M])
N2 = data3[:,:,0]
QA = data3[:,:,1]

Navg = np.reshape(N2,[T1//S,S,N])
Navg = np.mean(Navg,axis=1)
Navg = np.mean(Navg,axis=1)

ambient_temp = data[:, -1]
huddling = data[:,-2]
print(huddling.shape)
n = 10
# huddling = huddling.reshape(n, len(huddling)//n)
# huddling = huddling.mean(axis=0)
# print(huddling_means.shape)

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
# print(f"ambient_temp: {ambient_temp}")
# print(f"body_temp: {body_temp}")
# print(sorted_bt)

# temp_vec = np.linspace(10, 22, T)
Fig = plt.figure(figsize=(7, 7))
f1 = Fig.add_subplot(311)
f1.plot(ambient_temp, body_temp)
f1.set_xlabel('Ambient temperature')
f1.set_ylabel('Body temperature')
# for i, (x, y) in enumerate(zip(ambient_temp, body_temp)):
#     f1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 6), ha='center', va='center')

f2 = Fig.add_subplot(312)
f2.set_ylim(-0.5, 1.5)
# vector = np.linspace(1, 6000, 6000)
f2.plot(association)
f2.set_xlabel('Iteration')
f2.set_ylabel('Association')

f3 = Fig.add_subplot(313)
f3.set_title("Huddling")
# f3.set_ylim(0.2, 0.6)
days = np.linspace(1, 60, 12)
f3.plot(days, huddling, label="control")
# f3.plot(np.linspace(0,time[-1],Navg.shape[0]),Navg, label="learning")
f3.set_xlabel('Days')
f3.set_ylabel('Huddle Size')
f3.legend()
# f3.axis([0,60,1,np.ceil(N/2)+1])
# f3.set_aspect(np.diff(f3.get_xlim())/np.diff(f3.get_ylim()), adjustable='box')

plt.tight_layout()
plt.show()