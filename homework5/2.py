import numpy as np
from scipy.special import factorial
e = np.e

N = 100*100
qbar = 3
sigma = qbar

def poisson_pdf(qbar,k):
    return e**(-qbar) * qbar**k / factorial(k)

pdf = poisson_pdf(qbar, np.arange(5))


P_lessthan5 = np.sum(pdf)

npix = (1 - P_lessthan5) * N

print(f"npix is {npix}")
