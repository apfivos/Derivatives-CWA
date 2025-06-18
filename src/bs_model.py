from scipy.stats import norm
import numpy as np

def put_option(S0, K, T, r, q, sigma):
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return put_price

def call_option(S0, K, T, r, q, sigma):
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def put_call_parity(S0, K, T, r, q, put_price):
    call_price = S0 * np.exp(-q * T) - K * np.exp(-r * T) + put_price
    return call_price
