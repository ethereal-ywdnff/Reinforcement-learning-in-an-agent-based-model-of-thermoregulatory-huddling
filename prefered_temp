import numpy as np
import pylab as pl

t = np.linspace(0,60,12)

k = 8.31
p = np.exp(-t/k)
s = -k*p*np.log(p)
g = k*(1+s)
t1 = 8.*p
n = 19.*np.exp(-k*t1)
t2 = n*g/40.

tp = 36+t1-t2

F = pl.figure()
f = F.add_subplot(111)
f.plot(t,tp)
f.set_xlabel("time (days)")
f.set_ylabel("preferred temperature (degrees)")
pl.show()