import numpy as np

def european_put_binomial(S0, K, T, r, q, sigma, n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    q_prob = 1 - p

    stock_price = np.zeros((n + 1, n + 1))
    option_price = np.zeros((n + 1, n + 1))

    stock_price[0, 0] = S0
    for i in range(1, n + 1):
        stock_price[i, 0] = stock_price[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_price[i, j] = stock_price[i - 1, j - 1] * d

    for j in range(n + 1):
        option_price[n, j] = np.maximum(0, K - stock_price[n, j])

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_price[i, j] = np.exp(-r * dt) * (
                p * option_price[i + 1, j] + q_prob * option_price[i + 1, j + 1])

    return option_price[0, 0]

def binomial_prices_over_steps(S0, K, T, r, q, sigma, step_list):
    prices = []
    for n in step_list:
        prices.append(european_put_binomial(S0, K, T, r, q, sigma, n))
    return prices
