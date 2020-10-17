import numpy as np
import matplotlib.pyplot as plt

gamma = np.asarray(
                [0.1, 0.2, 0.4, 0.8] + ([1.0]*12)
                    )

sigma_d = np.asarray(
                [0.03, 0.05, 0.085, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32]
                    )
qmin = np.log10(10)
qmax = np.log10(10000)
q_A = np.logspace(qmin, qmax, 16)


DQE = (gamma * np.log10(np.e))**2 / (q_A * sigma_d**2)


fig = plt.figure()
ax = fig.add_subplot(111)
# ax.set_xticks(np.arange(qmin,qmax,.2))
plt.plot(q_A, DQE)
plt.xscale('log')
plt.yscale('log')

plt.title("problem7 DQE vs logE")
plt.ion()
plt.show()

import pdb; pdb.set_trace()
