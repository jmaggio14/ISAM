import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import cycle


k = 1


all_eta = {
        'screen':{
                'fast': 0.2,
                'slow': 0.1,
        },
        'film':{
                'fast': 0.1,
                'slow': 0.05,
        }

}

all_gamma = {
        'screen':{
                'fast': 0.7,
                'slow': 0.7,
        },
        'film':{
                'fast': 0.7,
                'slow': 0.7,
        }

}

all_qin = {
        'screen':{
                'fast': 5,
                'slow': 10,
        },
        'film':{
                'fast': 5,
                'slow': 10,
        }

}

all_H = {
        'screen':{
                'fast': 0.5,
                'slow': 0.05,
        },
        'film':{
                'fast': 0.05,
                'slow': 0.005,
        }

}


u = np.linspace(0,10,100)


def M(u,H):
    return 1 / (1 + H*u**2)

def Wdf(eta,gamma,qin):
    return eta * gamma * qin

def Wdscrn(u,eta,k,screen_H,film_H,qin):
    return (np.log10(np.e) * k)**2 / (eta**2 * qin) * (M(u,screen_H) * M(u,film_H))**2

##########
# Fast screen with fast and slow film

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('u (cyc/mm)')
plt.ylabel('W[u]')
plt.title("Fast Screen, W[u] by film type")


# fast screen
screen_eta = all_eta['screen']['fast']
film_etas = all_eta['film']

screen_gamma = all_gamma['screen']['fast']
film_gammas = all_gamma['film']

screen_qin = all_qin['screen']['fast']
film_qins = all_qin['film']

screen_H = all_H['screen']['fast']
film_Hs = all_H['film']

colors = ['C0','C1']


for f_type,c in zip(film_Hs.keys(), colors):
    film_H = film_Hs[f_type]
    film_eta = film_etas[f_type]
    film_gamma = film_gammas[f_type]
    film_qin = film_qins[f_type]

    Wsys = Wdscrn(u,screen_eta,k,screen_H,film_H,screen_qin) + Wdf(film_eta,film_gamma,film_qin)
    plt.plot(u, Wsys, color=c, label=f'film="{f_type}"')


plt.legend()




##########
# Slow screen with fast and slow film

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('u (cyc/mm)')
plt.ylabel('W[u]')
plt.title("Slow Screen, W[u] by film type")


# fast screen
screen_eta = all_eta['screen']['slow']
film_etas = all_eta['film']

screen_gamma = all_gamma['screen']['slow']
film_gammas = all_gamma['film']

screen_qin = all_qin['screen']['slow']
film_qins = all_qin['film']

screen_H = all_H['screen']['slow']
film_Hs = all_H['film']

colors = ['C0','C1']


for f_type,c in zip(film_Hs.keys(), colors):
    film_H = film_Hs[f_type]
    film_eta = film_etas[f_type]
    film_gamma = film_gammas[f_type]
    film_qin = film_qins[f_type]

    Wsys = Wdscrn(u,screen_eta,k,screen_H,film_H,screen_qin) + Wdf(film_eta,film_gamma,film_qin)
    plt.plot(u, Wsys, color=c, label=f'film="{f_type}"')


plt.legend()


plt.ion()
plt.show()


import pdb; pdb.set_trace()
