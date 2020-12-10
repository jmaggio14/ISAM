import numpy as np
import matplotlib.pyplot as plt



def M3(H,u):
    return 1 / (1 + H*(u**2))

def Map(ax,u):
    return ax * sinc(u)

def sinc(u):
    out = np.ones(u.shape)
    f = 2 * np.pi * u
    out[u!=0] = np.sin(f[u!=0]) / f[u!=0]
    return out

eta1 = 0.6
eta4 = 0.3
epsilon = 400
Gbar = 1000
ax = 50e-3 # mm
delta_x = 200e-3 # mm
H = 0.035 # mm^2
alpha = 1
Bx = 1

Un = 1 / (2 * delta_x)
C0 = 1

u = np.asarray([0, Un/2, Un])
q = 1

phi0 = 1
def W0(q,u):
    return q * np.ones(u.shape)

phi1 = eta1
def W1(q,u):
    return eta1 * q * np.ones(u.shape)

phi2 = phi1 * Gbar
def W2(q,u):
    return ((eta1 * Gbar**2 * q * (1 + epsilon/Gbar)) + (eta1 * Gbar * q) )* np.ones(u.shape)

phi3 = phi2
def W3(q,u):
    return (eta1 * Gbar**2 * q * (1 + epsilon/Gbar) * (M3(H,u)*M3(H,u))) + (eta1 * Gbar * q)

phi4 = phi3 * eta4
def W4(q,u):
    return (eta1 * eta4**2 * Gbar**2 * q * (1 + epsilon/Gbar) * M3(H,u)**2) + (eta1 * eta4 * Gbar * q)

phi5 = phi4 * Bx
def W5(q,u):
    return W4(q,u) * Bx**2

phi6 = phi5 * alpha
def W6(q,u):
    return (alpha**2 * ax**2 * Map(ax,u)**2) * W5(q,u)

phi7 = phi6
def W7(q,u):
    return W6(q,u) + np.sum( [W6(q,u+(i/delta_x)) for i in range(10)] )

def DQE(Win,Wout,phi,Msys):
    return (Win * phi**2 * Msys**2) / Wout

DQE1 = DQE( W0(q, u), W1(q, u), phi1, 1)

DQE2 = DQE( W1(q, u), W2(q, u), phi2, 1)

DQE3 = DQE( W2(q, u), W3(q, u), phi3, M3(H,u) )

DQE4 = DQE( W3(q, u), W4(q, u), phi4, M3(H,u) )

DQE5 = DQE( W4(q, u), W5(q, u), phi5, M3(H,u) )

DQE6 = DQE( W5(q, Bx*u), W6(q, Bx*u), phi6, M3(H,u) * Map(ax,Bx*u) )

DQE7 = DQE( W6(q, Bx*u), W7(q, Bx*u), phi7, M3(H,u) * Map(ax,Bx*u) )

print(f"DQE1={DQE1}")
print(f"DQE2={DQE2}")
print(f"DQE3={DQE3}")
print(f"DQE4={DQE4}")
print(f"DQE5={DQE5}")
print(f"DQE6={DQE6}")
print(f"DQE7={DQE7}")
