import numpy as np
from scipy.stats import norm

def bs_call(S, K, T, r, sigma, q=0.01):
    if T <= 0 or sigma <= 0:
        return np.nan
    
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def bs_call_vectorized(S, K, T, r, sigma, q=0.01):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    out = np.full_like(S, np.nan, dtype=float)
    valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)

    if not np.any(valid):
        return out

    Sv = S[valid]
    Kv = K[valid]
    Tv = T[valid]
    rv = r[valid]
    sigmav = sigma[valid]

    d1 = (np.log(Sv / Kv) + (rv - q + 0.5 * sigmav**2) * Tv) / (sigmav * np.sqrt(Tv))
    d2 = d1 - sigmav * np.sqrt(Tv)

    out[valid] = Sv * np.exp(-q * Tv) * norm.cdf(d1) - Kv * np.exp(-rv * Tv) * norm.cdf(d2)
    return out
