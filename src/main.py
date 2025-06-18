from bs_models import call_price, put_option, put_call_parity
from binomial import binomial_tree
from advanced_binomial import american_floating_strike_put_option
from monte_carlo import (
    monte_carlo_european_put,
    confidence_interval,
    simulate_interest_rate_paths,
    monte_carlo_with_stochastic_rates,
    down_and_out_put_option
)
from longstaff_schwartz import longstaff_schwartz_american_put

# Common parameters
S0 = 35
K = 35
T = 1
r = 0.03
q = 0.04
sigma = 0.45

# Task 1: Black-Scholes and Put-Call Parity
put = put_option(S0, K, T, r, q, sigma)
call = call_price(S0, K, T, r, q, sigma)
parity_call = put_call_parity(S0, K, T, r, q, put)
print("\nTask 1")
print("Put Option Price:", put)
print("Call Option Price:", call)
print("Call from Put-Call Parity:", parity_call)

# Task 2: Binomial Tree (European put)
print("\nTask 2")
binomial_price = binomial_tree(S0, K, T, r, q, sigma, 40000)
print("Binomial Tree Price (40,000 steps):", binomial_price)

# Task 3: American Floating Lookback Put
print("\nTask 3")
floating_put = american_floating_strike_put_option(S0, K, T, r, q, sigma, 25)
print("Floating Lookback Put (n=25):", floating_put)

# Task 4a: Monte Carlo European Put
print("\nTask 4a")
N = 10**6
price_mc, std_err = monte_carlo_european_put(S0, K, T, r, q, sigma, N)
conf_int = confidence_interval(price_mc, std_err)
print("Monte Carlo Price:", price_mc)
print("Standard Error:", std_err)
print("Confidence Interval:", conf_int)

# Task 4b: Monte Carlo with Stochastic Interest Rates
print("\nTask 4b")
rates = simulate_interest_rate_paths(0.01, 0.001, 0.05, 0.5, 252, N)
price_stoch, std_err_stoch = monte_carlo_with_stochastic_rates(S0, K, T, q, sigma, rates, N)
conf_int_stoch = confidence_interval(price_stoch, std_err_stoch, alpha=0.05)
print("Stochastic Rates Price:", price_stoch)
print("Standard Error:", std_err_stoch)
print("Confidence Interval:", conf_int_stoch)

# Task 4c: Down-and-Out Put Option
print("\nTask 4c")
B = 30
n_steps = 252
price_dop, std_err_dop = down_and_out_put_option(S0, K, T, r, q, sigma, B, N, n_steps)
conf_int_dop = confidence_interval(price_dop, std_err_dop, alpha=0.05)
print("Down-and-Out Put Price:", price_dop)
print("Standard Error:", std_err_dop)
print("Confidence Interval:", conf_int_dop)

# Task 7: Longstaff-Schwartz American Put
print("\nTask 7")
price_ls = longstaff_schwartz_american_put(S0, K, T, r, q, sigma, N=10**6, n_steps=252, degree=4)
print("Longstaff-Schwartz American Put Price:", price_ls)
