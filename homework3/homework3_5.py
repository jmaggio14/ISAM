import numpy as np
from scipy.special import factorial
e = np.e

def poisson_pmf(b,k):
    return e**(-b) * b**k / factorial(k)

pdf = poisson_pmf(0.25, np.arange(6))

import pdb; pdb.set_trace()
