import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Var:
    @classmethod
    def __iter__(cls):
        return (val for val in [cls.low, cls.center, cls.high])

class Epsilon(Var):
    low = 5
    center = 50
    high = 500

class G(Var):
    low = 10
    center = 100
    high = 1000

class Hscrn(Var):
    low = 0.035
    center = 0.35
    high = 3.5

u = np.linspace(0,10,100)
Hd = 0.035
eta1 = 0.6
eta2 = eta1

################################################################################
def M(H,u):
    return 1/(1 + H*(u**2))

def DQE(u, eta1, eta2, epsilon, G, Hscrn, Hd):
    Mscrn = M(Hscrn, u)
    Md = M(Hd, u)

    numerator = eta1
    denom = 1 + (epsilon / G) + (1 / (G * Mscrn**2 * Md**2 * eta2))
    return numerator / denom



################################################################################


fig = plt.figure()
fig.add_subplot(311)

# A
ax = plt.subplot(311)
plt.title("DQE for various values of Poisson Excess")
plt.ylim(0,1)
plt.axhline(eta1,linestyle=':',color='k')

for epsilon in Epsilon():
    dqe = DQE(u, eta1, eta2, epsilon, G.center, Hscrn.center, Hd)
    plt.plot(u, dqe, label=r'$\epsilon$='+f'{epsilon}')

handles, _ = plt.gca().get_legend_handles_labels()
handles.extend([Line2D([0], [0], color='k', linestyle=':', label=r'$DQE_{max}=\eta_{1}$')])
plt.legend(handles=handles)


# B
ax = plt.subplot(312)
plt.title("DQE for various mean gains")
plt.ylim(0,1)
plt.axhline(eta1,linestyle=':',color='k')

for g in G():
    dqe = DQE(u, eta1, eta2, Epsilon.center, g, Hscrn.center, Hd)
    plt.plot(u, dqe, label=r'$G$='+f'{g}')

handles, _ = plt.gca().get_legend_handles_labels()
handles.extend([Line2D([0], [0], color='k', linestyle=':', label=r'$DQE_{max}=\eta_{1}$')])
plt.legend(handles=handles)


# C
ax = plt.subplot(313)
plt.title("DQE for various values of Hscrn")
plt.ylim(0,1)
plt.axhline(eta1,linestyle=':',color='k')

for hscrn in Hscrn():
    dqe = DQE(u, eta1, eta2, Epsilon.center, G.center, hscrn, Hd)
    plt.plot(u, dqe, label=r'$H_{scrn}$='+f'{hscrn}')

handles, _ = plt.gca().get_legend_handles_labels()
handles.extend([Line2D([0], [0], color='k', linestyle=':', label=r'$DQE_{max}=\eta_{1}$')])
plt.legend(handles=handles)



plt.ion()
plt.show()
import pdb; pdb.set_trace()
