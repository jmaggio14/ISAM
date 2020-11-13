import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import cycle

colors = ['C' + str(i) for i in range(4)]
styles = ['-',':','--']


def DQE(qbar,eta,b1):
    return eta / (1 + (b1/(eta*qbar)))




qbar = np.linspace(1,100000,20000)
bs = list([1,10,100])
etas = list(reversed([0.1,0.3,0.5,0.7]))


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

axes=[ax1,ax2,ax3,ax4]


plt.suptitle('Prob14: Single Stage Amplifier DQE')

for eta,ax in zip(etas,axes):
    plt.sca(ax)
    plt.xscale('log')
    plt.ylim(0,1)
    plt.title(r'$\eta$'+f'={eta}')

    for b1,c in zip(bs,colors):
        print(f'b1={b1},eta={eta}')
        dqe = DQE(qbar,eta,b1)
        plt.plot(qbar,dqe,color=c,label=f'b1={b1}')

    plt.legend()


plt.sca(ax1)
plt.ylabel('DQE')

plt.sca(ax2)

plt.sca(ax3)
plt.ylabel('DQE')
plt.xlabel(r'$log(\bar{q})$')

plt.sca(ax4)
plt.xlabel(r'$log(\bar{q})$')

plt.ion()
plt.show()
import pdb; pdb.set_trace()

# b1_0 = Line2D([0], [0], color=c, linestyle=styles[0], label=f'b1={0}')
# b1_1 = Line2D([0], [0], color=c, linestyle=styles[1], label=f'b1={1}')
# b1_2 = Line2D([0], [0], color=c, linestyle=styles[2], label=f'b1={2}')
#
#
# eta_0 = Line2D([0], [0], color=colors[0], linestyle='-', label=r'$\eta$'+f'={etas[0]}')
# eta_1 = Line2D([0], [0], color=colors[1], linestyle='-', label=r'$\eta$'+f'={etas[1]}')
# eta_2 = Line2D([0], [0], color=colors[2], linestyle='-', label=r'$\eta$'+f'={etas[2]}')
# eta_3 = Line2D([0], [0], color=colors[3], linestyle='-', label=r'$\eta$'+f'={etas[3]}')
#
#
# handles = [b1_0,b1_1,b1_2,eta_0,eta_1,eta_2,eta_3]
# plt.legend(handles=handles)
