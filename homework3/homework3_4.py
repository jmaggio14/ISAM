import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial


def combo(N,k):
    return factorial(N) / (factorial(N-k) * factorial(k))

def bernoulli(p, N, k):
    combos = combo(N,k)
    return combos * (p**k) * (1 - p)**(N-k)


ks = np.arange(7)
print(ks)
pdf = bernoulli(0.4, 6, ks)
cdf = np.cumsum(pdf)
watts = ks * 0.5

fig = plt.figure()
ax = fig.add_subplot(211)
plt.title("pdf of room intercom power draw (problem 4)")
plt.xlabel("watts")
plt.ylabel("pdf")
plt.plot(watts, pdf)


ax = fig.add_subplot(212)
plt.title("cdf of power draw (problem 4)")
plt.xlabel("watts")
plt.ylabel("cdf")
plt.plot(watts, cdf)

print(pdf[0])
print(1-cdf[np.argmax(watts >= 2)])


plt.ion()
plt.show()
import pdb; pdb.set_trace()
