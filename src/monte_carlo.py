import numpy as np
import scipy.stats as ss
from scipy.stats.qmc import Halton

# Task 4a - European Put Option (standard Monte Carlo with Halton)
def monte_carlo_european_put(S0, K, T, r, q, sigma, n):
    h_sampler = Halton(d=1, scramble=True, seed=7)
    h_samples = h_sampler.random(n)
    z = ss.norm.ppf(h_samples)

    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * z.flatten()
    final_prices = S0 * np.exp(drift + diffusion)

    payoffs = np.exp(-r * T) * np.maximum(K - final_prices, 0)
    price = np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n)

    return price, std_error

def confidence_interval(price, std_error, alpha=0.01):
    z_score = ss.norm.ppf(1 - alpha / 2)
    return price - z_score * std_error, price + z_score * std_error

# Task 4b - Monte Carlo with Stochastic Interest Rate Paths
def simulate_interest_rate_paths(r0, alpha, beta, rho, steps, n):
    dt = 1 / steps
    h_sampler = Halton(d=2, scramble=True, seed=7)
    h_samples = h_sampler.random(n * steps).reshape(n, steps, 2)

    Z_stock = ss.norm.ppf(h_samples[:, :, 0])
    Z_ir = ss.norm.ppf(h_samples[:, :, 1])

    Z_ir_corr = rho * Z_stock + np.sqrt(1 - rho ** 2) * Z_ir

    rates = np.zeros((n, steps + 1))
    rates[:, 0] = r0

    for t in range(1, steps + 1):
        rates[:, t] = rates[:, t - 1] * np.exp(
            (alpha - 0.5 * beta ** 2) * dt + beta * np.sqrt(dt) * Z_ir_corr[:, t - 1]
        )

    return rates

def monte_carlo_with_stochastic_rates(S0, K, T, q, sigma, rates, n):
    dt = 1 / rates.shape[1]
    rT = rates[:, -1]
    disc_factors = np.exp(-np.sum(rates[:, 1:] * (T / 252), axis=1))

    h_sampler = Halton(d=1, scramble=True, seed=7)
    z = ss.norm.ppf(h_sampler.random(n))

    drift = (rT - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * z.flatten()
    final_prices = S0 * np.exp(drift + diffusion)

    payoffs = disc_factors * np.maximum(K - final_prices, 0)
    price = np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n)

    return price, std_error

# Task 4c - Down-and-Out Put Option (Barrier Option)
def down_and_out_put_option(S0, K, T, r, q, sigma, B, n_paths, n_steps):
    dt = 1 / n_steps
    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S0

    h_sampler = Halton(d=1, scramble=True, seed=7)
    Z = ss.norm.ppf(h_sampler.random(n_paths * n_steps).reshape(n_paths, n_steps))

    active_paths = np.ones(n_paths, dtype=bool)
    for i in range(1, n_steps + 1):
        S_paths[active_paths, i] = S_paths[active_paths, i - 1] * np.exp(
            (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[active_paths, i - 1]
        )
        active_paths &= S_paths[:, i] >= B

    payoffs = np.maximum(K - S_paths[:, -1], 0)
    payoffs[~active_paths] = 0
    price = np.mean(payoffs * np.exp(-r * T))
    std_error = np.std(payoffs) / np.sqrt(n_paths)

    return price, std_error
