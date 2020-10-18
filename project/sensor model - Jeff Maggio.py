import numpy as np
from scipy.stats import poisson
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("tkagg")
################################################################################
class SensorModel(object):
    def __init__(self, L, eta, read_noise, bitdepth):
        """Instantiates the sensor model

        Args:
            L(int): full well capacity of the detector elements [e-]
            eta(float,np.ndarray[1,N]): primary quantum efficiency
                (electron/photon efficiency of photon absorption on sensor)
                  must be between 0 and 1
            read_noise(int): read noise of ADC [e-]

        """
        self.L = L
        self.eta = eta
        self.read_noise = read_noise
        self.bitdepth = bitdepth

    def f1(self, qbar):
        """f1 is the sum of the mean remaining capacities for every value of k

        Args:
            qbar(np.ndarray[Nx1], float): mean number of photons per detector element
                (poisson mean)
        """
        ks = np.arange(self.L)
        kk,qq = np.meshgrid(ks,qbar)

        val = np.sum(poisson.cdf(kk,qq),axis=1)

        return val / self.L

    # --------------------------------------------------------------------------
    def f2(self, qbar):
        """
        Args:
            qbar(np.ndarray[Nx1], float): mean number of photons per detector element
                (poisson mean)

        """
        val = poisson.cdf(self.L,qbar)
        # val = 0
        # for i in range(self.L):
        #     val += poisson.pmf(i,qbar) / self.L

        # import pdb; pdb.set_trace()
        return val / self.L

    # --------------------------------------------------------------------------
    def f3(self, qbar):
        """
        Args:
            qbar(np.ndarray[Nx1], float): mean number of photons per detector element
                (poisson mean)

        """
        ks = np.arange(self.L)#.reshape( (self.L,1) )
        coeffs = np.linspace(0, (2*self.L)-1, self.L)#.reshape( (1,self.L) )


        qq,kk = np.meshgrid(qbar,ks)
        _,cc = np.meshgrid(qbar,coeffs)

        val = np.sum(cc * poisson.cdf(kk,qq),axis=0)

        return val / (self.L**2)

    # --------------------------------------------------------------------------
    def ideal_DQE(self, qbar):
        """
        Calculates DQE as
            \frac{\bar{q} (f2)^{2}}{(1-f3)-(1-f1)^{2}}

        Args:
            qbar(np.ndarray, float): mean number of photons per detector element
                (poisson mean)

        Returns:
            DQE(np.ndarray, float): DQE for the given value(s) of qbar
        """
        f1 = self.f1(qbar)
        f2 = self.f2(qbar)
        f3 = self.f3(qbar)
        ideal_DQE = (qbar * (f2**2)) / ((1-f3) - (1-f1)**2)
        return ideal_DQE

    def ideal_components(self,qbar):
        f1 = self.f1(qbar)
        f2 = self.f2(qbar)
        f3 = self.f3(qbar)
        SN_in = (qbar * (f2**2))
        sigma_sq_l = ((1-f3) - (1-f1)**2)
        return SN_in, sigma_sq_l

    # --------------------------------------------------------------------------
    def DQE(self, qbar, a2d_noise=True):
        ideal_SN_in, sigma_sq_l = self.ideal_components(qbar)

        sigma_sq_r = self.read_noise / self.L**2
        sigma_sq_a2d = self.a2d_noise() if a2d_noise else 0

        ideal_DQE = ideal_SN_in / sigma_sq_l
        DQE = ideal_SN_in / (sigma_sq_l + sigma_sq_r + sigma_sq_a2d)
        # import pdb; pdb.set_trace()
        return self.eta * DQE

    # --------------------------------------------------------------------------
    def poisson_noise(self,qbar):
        f1 = self.f1(qbar)
        f3 = self.f3(qbar)
        return ((1-f3) - (1-f1)**2) * (self.L**2)

    # --------------------------------------------------------------------------
    def a2d_noise(self):
        N = int(2**self.bitdepth) - 1
        return (2*N)**2 / 12 / self.L**2

    # --------------------------------------------------------------------------
    def quantize(self, arr):
        # N = int(2**self.bitdepth)
        # # quant = (arr / N)
        # bins = np.linspace(0, self.L, N)
        # indices = np.digitize(arr, bins, right=False)
        # return bins[indices]
        return arr

################################################################################



# Problem 1
################################################################################
L = 1024
read_noise = 10
bitdepth = 4
etas = [0.125, 0.25, 0.5, 1.0]

qbar = np.arange(1,8001,200)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.title("DQE vs mean number of photons")
plt.ylabel("DQE")
plt.xlabel("qbar / eta")
# plt.ylim(0,1)
plt.xlim(0,1e4)

plt.ion()
plt.show()
plt.legend()

dqes = []
for eta in reversed(etas):
    sensor = SensorModel(L, eta, read_noise, bitdepth)
    dqe = sensor.DQE(qbar) * eta
    dqes.append(dqe)

    plt.plot(qbar / eta, dqe, label=f"eta={eta}")

    print(f"eta is {eta}")
    print(f"f1 is {np.round(sensor.f1(qbar),5)[:11]}")
    print(f"f3 is {np.round(sensor.f3(qbar),5)[:11]}")
    print(f"f2 is {np.round(sensor.f2(qbar),5)[:11]}")
    print()
    # import pdb; pdb.set_trace()

plt.legend()


# import pdb; pdb.set_trace()
# the DQE is improperly scaled
# f2 is off by approximately a factor of 1000

# Problem 2
################################################################################


# part A
# ----------
L = 1024
eta = 0.5
read_noises = [1,3,10]
bitdepths = 4
qbar = np.arange(1,3001,200)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title("DQE vs mean number of photons, (no a2d noise)")
plt.ylabel("DQE")
plt.xlabel("qbar / eta")
# plt.ylim(0,1)
plt.xlim(0,3001)

plt.ion()
plt.show()

for rn in read_noises:
    sensor = SensorModel(L, eta, rn, bitdepth)
    dqe = sensor.DQE(qbar, a2d_noise=False)

    plt.plot(qbar / eta, dqe, label=f"read noise={rn}")

plt.legend()

import pdb; pdb.set_trace()
