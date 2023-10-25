import numpy as np


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