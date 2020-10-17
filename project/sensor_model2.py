
# --------------------------------------------------------------------------
def f1(qbar):
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
def f2(qbar):
    """
    Args:
        qbar(np.ndarray[Nx1], float): mean number of photons per detector element
            (poisson mean)

    """
    val = poisson.cdf(L,qbar)

    return val / self.L

# --------------------------------------------------------------------------
def f3(qbar):
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
