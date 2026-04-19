from scipy.stats import norm
import numpy as np

def compute_greeks(df):
    eps = 1e-6
    
    S = df["S"].values
    K = df["K"].values
    T = df["T"].values
    r = df["r"].values
    sigma = df["sigma"].values

    sigma = np.maximum(sigma, eps)
    T = np.maximum(T, eps)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    df["delta"] = norm.cdf(d1)
    df["gamma"] = norm.pdf(d1) / (S * sigma * np.sqrt(T) + eps)
    df["vega"] = S * norm.pdf(d1) * np.sqrt(T)
    df["theta"] = (
        - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )

    return df

def recompute_features(df):
    eps = 1e-6

    # basic
    df["sqrt_T"] = np.sqrt(df["T"])
    df["sigma_sqrt_T"] = df["sigma"] * df["sqrt_T"]

    # log moneyness
    df["log_moneyness"] = np.log(df["S"] / df["K"])

    # Greeks
    df = compute_greeks(df)

    # derived
    df["vix_ratio"] = df["sigma"] / (df["vix"] + eps)
    df["time_vol"] = df["sigma"] * np.sqrt(df["T"])
    df["vega_scaled"] = df["vega"] / (df["Market_Price"] + eps)

    return df