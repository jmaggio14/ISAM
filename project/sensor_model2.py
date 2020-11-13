import numpy as np
from scipy.stats import poisson
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 15})

QBAR = np.linspace(1,8001,400)
L = 1024

################################################################################
def calc_f1(qbar,L):
    val = np.e**(-qbar)
    for k in range(1,L):
        val += poisson.cdf(k,qbar)

    return val / L

################################################################################
def calc_f2(qbar,L):
    val = poisson.cdf(L-1,qbar)
    return val / L

################################################################################
def calc_f3(qbar,L):
    val = np.e**(-qbar)
    for k in range(1,L):
        coeff = ((2*k) - 1) + 2
        val += coeff * poisson.cdf(k,qbar)

    return val / (L**2)

################################################################################
# IDEAL DQE
f1 = calc_f1(QBAR,L)
f2 = calc_f2(QBAR,L)
f3 = calc_f3(QBAR,L)
NUMERATOR = (QBAR * (f2**2)) * L**2
sigma_sq_l = ( (1-f3) - (1-f1)**2 ) * L**2
ideal_DQE =  NUMERATOR / sigma_sq_l

# plot ideal DQE
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title("Ideal DQE (shot noise only)",)
plt.xlabel(r"$\bar{q}$ / $\eta$ $\it{photons}$",)
plt.ylabel("DQE",)
plt.xticks(fontsize='x-large')
plt.yticks(fontsize='x-large')
plt.axvline(L,label='L (saturation)',linestyle=':',linewidth=3)
plt.plot(QBAR,ideal_DQE,label=r"ideal DQE ($\eta$=1)",linewidth=4)
plt.legend()

plt.ion()
plt.show()
plt.savefig('out/ideal_DQE.png')


################################################################################
#                                       1
################################################################################
# DQE with eta included
etas = [0.125, 0.25, 0.5, 1.0]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title(r"DQE scales with primary quantum efficiency")
plt.xlabel(r"$\bar{q}$ / $\eta$ $\it{[photons]}$")
plt.ylabel("DQE")

for eta in etas:
    dqe_eta = ideal_DQE * eta
    scaled_q = QBAR / eta
    lines = plt.plot(scaled_q, dqe_eta, label=r'$\eta$='+f'{eta}')
    plt.axvline(L/eta,linestyle=':',color=lines[-1].get_color())


handles, _ = plt.gca().get_legend_handles_labels()
handles.extend([Line2D([0], [0], color='k', linestyle=':', label='saturation')])
plt.legend(handles=handles)
plt.ion()
plt.show()
plt.savefig('out/eta_scaled_DQE.png')

################################################################################
# DQE with read noise = 10 and 4bit AD noise
bitdepth = 4
read_noise = 10
sigma_r_sq = read_noise**2
sigma_ad_sq = (L**2) / (12 * 2**(2*bitdepth))

fig = plt.figure()
ax1 = fig.add_subplot(111)
fig.suptitle(r"All three DQE aberrations together")
# ax1.set_title("no quantization")
plt.xlabel(r"$\bar{q}$ / $\eta$ $\it{photons}$")
plt.ylabel("DQE")
plt.xlim(0,1e4)
plt.ylim(0,1.05)

for eta in etas:
    scaled_q = QBAR / eta
    dqe = eta * NUMERATOR / (sigma_sq_l + sigma_r_sq + sigma_ad_sq)
    lines = plt.plot(scaled_q, dqe, label=r'$\eta$='+f'{eta}')
    plt.axvline(L/eta,linestyle=':',color=lines[-1].get_color())

handles, _ = plt.gca().get_legend_handles_labels()
handles.extend([Line2D([0], [0], color='k', linestyle=':', label='saturation')])
plt.legend(handles=handles)
plt.ion()
plt.show()
plt.savefig('out/read_noise_4bit_DQE.png')

################################################################################
#                                   2
################################################################################
# DQE with eta=0.5 and 0 AD noise, various read noises
eta = 0.5
read_noises = [1,3,10]
QBAR = np.arange(1,3001,200)
scaled_q = QBAR / eta


fig = plt.figure()
fig.suptitle(r"DQE shrinks in early region with increased read noise")
ax1 = fig.add_subplot(111)
# ax1.set_title("no quantization")
plt.xlabel(r"$\bar{q}$ / $\eta$ $\it{photons}$")
plt.ylabel("DQE")
plt.xlim(0,3000)
plt.ylim(0,1.05)

f1 = calc_f1(QBAR,L)
f2 = calc_f2(QBAR,L)
f3 = calc_f3(QBAR,L)
NUMERATOR = (QBAR * (f2**2)) * L**2
sigma_sq_l = ( (1-f3) - (1-f1)**2 ) * L**2

for rn in read_noises:
    dqe = eta * NUMERATOR / (sigma_sq_l + rn**2)
    sigma_ad_sq = (L**2) / (12 * 2**(2*bitdepth))
    lines = plt.plot(scaled_q, dqe, label=r'$\sigma_{read}^{2}$='+f'{rn}')

plt.axvline(L/eta,linestyle=':',color='k',label='saturation')
plt.axhline(eta,linestyle='--',alpha=0.5,color='red',label=r'$\eta$')
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles)
plt.ion()
plt.show()
plt.savefig('out/DQE_by_read_noise.png')

################################################################################
# DQE with eta=0.5 and no read noise, various AD bitdepths
bitdepths = [8,4,2]

fig = plt.figure()
fig.suptitle(r"DQE by bitdepth")
ax1 = fig.add_subplot(111)
# ax1.set_title("no quantization")
plt.xlabel(r"$\bar{q}$ / $\eta$ $\it{photons}$")
plt.ylabel("DQE")
plt.xlim(0,3000)
plt.ylim(0,1.05)

for b in bitdepths:
    sigma_ad_sq = (L**2) / (12 * 2**(2*b))
    dqe = eta * NUMERATOR / (sigma_sq_l + sigma_ad_sq)
    lines = plt.plot(scaled_q, dqe, label=f'{b} bits')

plt.axvline(L/eta,linestyle=':',color='k',label='saturation')
plt.axhline(eta,linestyle='--',alpha=0.5,color='red',label=r'$\eta$')
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles)
plt.ion()
plt.show()
plt.savefig('out/DQE_by_bitdepth.png')


################################################################################
#                                    3
################################################################################
# normalized pixel values vs logE for various pitch pitches

QBAR = np.linspace(1,3001,200)
f1 = calc_f1(QBAR,L)
pitches = [20,10,5]

fig = plt.figure()
fig.suptitle(r"Normalized Pixel Value vs Exposure by pixel pitch")
ax1 = fig.add_subplot(111)
# ax1.set_title("no quantization")
plt.xlabel(r"$log_{10}E$ [photons/400$\mu m^{2}$]")
plt.ylabel("l/L (normalized mean pixel value)")
plt.ylim(0,1.05)


# mean photons / px
l = L * (1 - f1)
normalized_l = l / L

E = QBAR
for p in pitches:
    plt.plot(np.log10(E),normalized_l,label=f'{p}'+r'$\mu m$')
    E = E * 4

plt.legend()
plt.ion()
plt.show()
plt.savefig('out/normalized_vs_logE.png')



################################################################################
#                                     4
################################################################################
#
pitches = [20,10,5]
QBAR = np.linspace(1,3001,200)
f1 = calc_f1(QBAR,L)
f2 = calc_f2(QBAR,L)
f3 = calc_f3(QBAR,L)
NUMERATOR = (QBAR * (f2**2)) * L**2
sigma_sq_l = ( (1-f3) - (1-f1)**2 ) * L**2

fig = plt.figure()
fig.suptitle(r"Variance vs Exposure by pixel pitch")
ax1 = fig.add_subplot(121)
ax1.set_title("Variance vs Normalized Mean Count")
# ax1.set_title("no quantization")
plt.xlabel("normalized mean count level")
plt.ylabel("Variance")

ax2 = fig.add_subplot(122)
ax2.set_title("Variance vs LogE")
plt.xlabel(r"$log_{10}E$ [photons/400$\mu m^{2}$]")
plt.ylabel("Variance")

l = L * (1 - f1)
normalized_l = l / L
E = QBAR
variance = sigma_sq_l

for p in pitches:
    A = p*p

    fig.add_subplot(121)
    plt.plot(normalized_l, variance, label=f'{p}'+r'$\mu m$')

    fig.add_subplot(122)
    plt.plot(np.log10(E), variance, label=f'{p}'+r'$\mu m$')

    variance = variance / 4
    E = E * 4

plt.legend()
plt.ion()
plt.show()
plt.savefig('out/variance_vs_normalized.png')

################################################################################
#                                     5
################################################################################
pitches = [20,10,5]
QBAR = np.linspace(0,3001,200)
f1 = calc_f1(QBAR,L)
f2 = calc_f2(QBAR,L)
f3 = calc_f3(QBAR,L)
NUMERATOR = (QBAR * (f2**2)) * L**2
sigma_sq_l = ( (1-f3) - (1-f1)**2 ) * L**2

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("DQE vs logE by pixel pitch")
# ax1.set_title("no quantization")
plt.xlabel(r"$log_{10}E$ [photons/400$\mu m^{2}$]")
plt.ylabel("DQE")



l = L * (1 - f1)
normalized_l = l / L
E = QBAR
variance = sigma_sq_l

for p in pitches:
    fig.add_subplot(111)
    dqe = NUMERATOR / sigma_sq_l
    plt.plot(np.log10(E), dqe, label=f'{p}'+r'$\mu m$')

    variance = variance / 4
    E = E * 4

plt.legend()
plt.ion()
plt.show()
plt.savefig('out/dqe_vs_logE.png')


################################################################################

fig = plt.figure()
fig.add_subplot(311)
fig.suptitle("DQE mean count level vs logE (mean # photons/400$\mu m^{2}$)")
QBAR = np.linspace(0,3001,200)

ax.set_ylabel("DQE")
E = QBAR
colors = ['C0','C1','C2']

for i,p in enumerate(pitches):
    ax = fig.add_subplot(3,1,i+1)
    ax.set_xlabel(r"$log_{10}E$ [photons/400$\mu m^{2}$]")
    ax.set_ylabel("DQE")
    ax.set_ylim(0,2)
    ax.set_title(f'{p}'+r'$\mu m$')
    ax.set_xlim(0,5)
    dqe = NUMERATOR / sigma_sq_l
    plt.plot(np.log10(E), dqe, color=colors[i], label=f'{p}'+r'$\mu m$')
    handles1, _ = ax.get_legend_handles_labels()

    ax_twin = ax.twinx()
    ax_twin.set_ylabel("normalized count")
    ax_twin.set_ylim(0,1)
    ax_twin.tick_params(axis='y')
    plt.plot(np.log10(E), normalized_l, color=colors[i], linestyle=':', label='mean count')
    handles2, _ = ax_twin.get_legend_handles_labels()
    plt.legend(handles=handles1+handles2, loc='upper left')

    E = E * 4


plt.ion()
plt.show()
plt.savefig('out/DQE_by_mean_count.png')


import pdb; pdb.set_trace()
