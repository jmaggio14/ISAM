import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

raw_data = np.asarray([[1,   0.0635],
                        [3,   0.0599],
                        [6,   0.0549],
                        [20,  0.0450],
                        [40,  0.0451],
                        [100, 0.0349],
                        [323, 0.0313]])

a_scaled = raw_data.copy()
a_scaled[:,0] = a_scaled[:,0] ** (-1/2)

def selwyn(A, G):
    return G / np.sqrt(2 * A)


# calculating G using a single data point
G = np.mean(a_scaled[1,1] * np.sqrt(2 * a_scaled[0,0]))
# G = np.mean(a_scaled[:,1] * np.sqrt(2 * a_scaled[:,0]))

selwyn_curve = selwyn(raw_data[:,0], G) + raw_data[-1,1]

# creating the figure
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(a_scaled[:,0], a_scaled[:,1], color='blue', label='raw')
plt.plot(a_scaled[:,0], selwyn_curve, color='red', label='basic selwyn')


# fit the curve
# sigma_d = c * A ** (-1 / n)
def modifed_selwyn(A, c, n):
    return c * A**(-1 / n)

A = raw_data[:,0]
sigma_d = raw_data[:,1]

(c, n), cov_mat = optimize.curve_fit(modifed_selwyn,
                            xdata=raw_data[:,0],
                            ydata=raw_data[:,1])

fitted = modifed_selwyn(raw_data[:,0],c,n)
plt.plot(A**-.5, fitted, color='g', label='fitted modified selwyn')
print(f"c is {c}\nn is {n}")

plt.xlabel('A^1/2')
plt.ylabel('sigma_d')
plt.legend()
plt.ion()
plt.show()



import pdb; pdb.set_trace()
