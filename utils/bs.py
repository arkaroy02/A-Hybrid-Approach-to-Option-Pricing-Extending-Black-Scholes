import numpy as np
from scipy.stats import norm

def bs_call(S, K, T, r, sigma, q=0.01):
    if T <= 0 or sigma <= 0:
        return np.nan
    
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)