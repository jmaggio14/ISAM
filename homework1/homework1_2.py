import imagepypelines as ip
import numpy as np


@ip.blockify(kwargs={'sigma':4})
def lsf(x, sigma):
    """generates a line spread function with gaussian profile"""
    return (1 / (sigma * np.sqrt(2 * np.pi) )) * np.e**(-x**2 / (2*sigma**2))


@ip.blockify()
def convolve1d(arr1, arr2):
    return np.convolve(arr1, arr2, mode='same')

@ip.blockify()
def image1(x):
    y = np.zeros(x.shape)
    y[ x < 50] = 0.1
    y[ x >= 50] = 1.0
    return y

@ip.blockify(kwargs={'a':0.5})
def image2(x, b, a):
    return a * (1 + np.sin(2*np.pi*b*x) )


tasks = {
        # inputs
        'f[x]_range' : ip.Input(0),
        'lsf_range'  : ip.Input(1),
        'b'          : ip.Input(2),
        # processing
        'l[x]'  : (lsf, 'lsf_range'),
        'f[x]_1' : (image1, 'f[x]_range'),
        'f[x]_2' : (image2, 'f[x]_range', 'b'),

        'g[x]_1'  : (convolve1d, 'f[x]_1', 'l[x]'),
        'g[x]_2'  : (convolve1d, 'f[x]_2', 'l[x]'),
        }


pipeline = ip.Pipeline(tasks)

lsf_range = np.arange(-10,11,1).astype(np.float64)
image_range = np.arange(0,101,1).astype(np.float64)



outs = []
for b in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]:
    out = pipeline.process([image_range], [lsf_range], [b])
    f1,g1,f2,g2 = out['f[x]_1'], out['g[x]_1'], out['f[x]_2'], out['g[x]_2']
    outs.append( [f1,g1,f2,g2,b] )


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


red = mpatches.Patch(color='r', label='g[x]')
blue = mpatches.Patch(color='b', label='f[x]')

nrows = len(outs) + 1
ncols = 2
nplots = 0

fig = plt.figure()
fig.legend(handles=[blue,red], loc='lower center')
plt.suptitle('effect of gaussian lsf on sine and step functions\n (first and last 10 g[x] data points exhibit boundary overlap error)')

ax = plt.subplot(nrows, ncols, 1)
ax.set_title('step function')

ax = plt.subplot(nrows, ncols, 2)
ax.set_title('sine')

for i,out in enumerate(outs):
    nplots += 1
    ax = plt.subplot(nrows, ncols, nplots)
    plt.plot(image_range, np.asarray(out[0]).flatten(), 'b')
    plt.plot(image_range, np.asarray(out[1]).flatten(), 'r')

    nplots += 1
    ax = plt.subplot(nrows, ncols, nplots)
    ax.set_ylabel(f'b={out[4]}')
    plt.plot(image_range, np.asarray(out[2]).flatten(), 'b')
    plt.plot(image_range, np.asarray(out[3]).flatten(), 'r')

plt.ion()
plt.show()


import pdb; pdb.set_trace()
