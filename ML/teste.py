import numpy as np
import matplotlib.pyplot as plt

x=[]
y=[]
i=0
z=[]
while (len(z) <= 10) and (i<100000):
    x.append(i)
    y.append(np.cos(0.001*i-0.5))
    z.append(np.sin(0.001*i-0.5))
    i+=1

plt.plot(y,z)
plt.savefig("testando123.png")