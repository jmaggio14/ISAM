import numpy as np
import matplotlib.pyplot as plt

def lsf(x, sigma):
    """generates a line spread function with gaussian profile"""
    return (1 / (sigma * np.sqrt(2 * np.pi) )) * np.e**(-x**2 / (2*sigma**2))


u = np.linspace(0,12,100)


mtf = lsf(u, 1) - lsf(10, 1)


print( lsf(2, 1) )
print( lsf(4, 1) )
print( lsf(10, 1) )


print( (lsf(2,1) - lsf(10, 1))**2 )
print( (lsf(4,1) - lsf(10, 1))**2 )
