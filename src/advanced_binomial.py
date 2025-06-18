import numpy as np

def _valuation_recursive(p, q, i, j, n, maxs, r, dt, stock_price):
    if i == n:
        return max(0, maxs - stock_price[i, j])

    new_max = max(maxs, stock_price[i, j])

    option_value = np.exp(-r * dt) * (
        p * _valuation_recursive(p, q, i + 1, j, n, new_max, r, dt, stock_price) +
        q * _valuation_recursive(p, q, i + 1, j + 1, n, new_max, r, dt, stock_price)
    )

    exercise_value = max(0, new_max - stock_price[i, j])
    return max(option_value, exercise_value)

def american_floating_lookback_put(S0, K, T, r, q, sigma, n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    q_prob = 1 - p

    stock_price = np.zeros((n + 1, n + 1))
    stock_price[0, 0] = S0

    for i in range(1, n + 1):
        stock_price[i, 0] = stock_price[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_price[i, j] = stock_price[i - 1, j - 1] * d

    option_price = _valuation_recursive(p, q_prob, 0, 0, n, S0, r, dt, stock_price)
    return option_price

def floating_lookback_prices(S0, K, T, r, q, sigma, n_range):
    prices = []
    for n in n_range:
        prices.append(american_floating_lookback_put(S0, K, T, r, q, sigma, n))
    return prices
