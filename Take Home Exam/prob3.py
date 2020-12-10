import numpy as np
import matplotlib.pyplot as plt



def M(H,u):
    return 1 / (1 + H*(u**2))

def sinc(u):
    out = np.ones(u.shape)
    f = 2 * np.pi * u[1:]
    out[1:] = np.sin(f) / f
    out[0] = 1
    return out

eta1 = 0.6
eta4 = 0.6
epsilon = 400
Gbar = 1000
ax = 50e-3 # um
delta_x = 200e-3 # um
H = 3.5 # mm^2
alpha = 1

Un = 1 / (2 * delta_x)
C0 = 1

u = np.asarray([0, Un/2, Un])

C1 = eta1
DQE1 = 1 / (1/C1 )
# ----------------------------
C2 = C1 * Gbar
DQE2 = 1 / (1/C1\
        + 1/(1 + 1/C1 + (1+epsilon)/C2)\
        )

# ----------------------------
M3 = M(H,u)
C3 = C2 * M3**2
DQE3 = 1 / (1/C1\
        + 1/(1 + 1/C1 + (1+epsilon)/C2)\
        + (1 - M3**2) / C3
        )

# ----------------------------
C4 = C3 * eta4
DQE4 = 1 / (1/C1\
        + 1/(1 + 1/C1 + (1+epsilon)/C2)\
        + (1 - M3**2) / C3\
        + (1-eta4) / C4
        )

# ----------------------------
M5 = ax * sinc(ax * u)
C5 = C4 * alpha * M5**2
DQE5 = 1 / (1/C1\
        + 1/(1 + 1/C1 + (1+epsilon)/C2)\
        + (1 - M3**2) / C3\
        + (1-eta4) / C4\
        + (1 - M5**2) / C5
        )

# ----------------------------
M6 = delta_x * sinc(delta_x * u)
C6 = C5 * M6**2
DQE6 = 1 / (1/C1\
        + 1/(1 + 1/C1 + (1+epsilon)/C2)\
        + (1 - M3**2) / C3\
        + (1-eta4) / C4\
        + (1 - M5**2) / C5
        # I'm unsure about this line
        + (1 - M6**2) / C6
        )

plt.figure()
plt.title("DQE by stage")
plt.xlabel("frequency")
plt.ylabel("DQE")
for i in range(1,7):
    key = f'DQE{i}'
    dqe = globals()[key]
    if not isinstance(dqe,np.ndarray):
        dqe = np.ones(u.shape) * dqe

    plt.plot(u, dqe, label=key)
    print(f"{key} : {list(np.round(dqe,5))}")

plt.xlim(0,3)
plt.axvline(Un,linestyle=':',label=r'$u_{N}$')
plt.axvline(Un/2,linestyle=':',label=r'$\frac{u_{N}}{2}$')
plt.ylim(0,1.1)
plt.legend()

# ==========================================================
plt.figure()
plt.title("Quantum accounting diagram for Problem 2")
plt.xlabel("stage")
plt.ylabel("quanta")
qads0 = []
qadsUn2 = []
qadsUn = []

for i in range(7):
    key = f'C{i}'
    qad = globals()[key]
    if not isinstance(qad, np.ndarray):
        qad = np.ones(u.shape) * qad

    qads0.append(round(qad[0],4))
    qadsUn2.append(round(qad[1],4))
    qadsUn.append(round(qad[2],4))

    print(f"{key} : {list(np.round(qad,5))}")


plt.plot(qads0, label='u=0')
plt.plot(qadsUn2, label=r'$u=\frac{u_{N}}{2}$')
plt.plot(qadsUn, label=r'$u=u_{N}$')
plt.legend()
plt.xlim(0,6.1)
plt.ion()
plt.yscale('log')
plt.show()


import pdb; pdb.set_trace()
