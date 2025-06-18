import numpy as np
import scipy.stats as ss


def longstaff_schwartz_american_put(S0, K, T, r, q, sigma, n_steps, n_paths, degree=4, seed=3):
    np.random.seed(seed)
    dt = T / n_steps
    df = np.exp(-r * dt)

    # Simulate price paths
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    W = ss.norm.rvs(drift, diffusion, size=(n_paths, n_steps - 1))

    X = np.concatenate((np.zeros((n_paths, 1)), W), axis=1).cumsum(axis=1)
    S = S0 * np.exp(X)

    # Initialize payoff matrix
    H = np.maximum(K - S, 0)
    V = np.zeros_like(H)
    V[:, -1] = H[:, -1]

    # Backward induction
    for t in range(n_steps - 2, 0, -1):
        good = H[:, t] > 0
        if not np.any(good):
            continue

        X_t = S[good, t]
        Y_t = V[good, t + 1] * df

        coeffs = np.polyfit(X_t, Y_t, deg=degree)
        C = np.polyval(coeffs, X_t)

        exercise = H[good, t] > C
        exercise_indices = np.where(good)[0][exercise]

        V[exercise_indices, t] = H[exercise_indices, t]
        V[exercise_indices, t + 1:] = 0

        # Discount continuation value where not exercised
        non_exercise_indices = np.setdiff1d(np.where(good)[0], exercise_indices)
        V[non_exercise_indices, t] = V[non_exercise_indices, t + 1] * df

    # Discount from time 1
    price = np.mean(V[:, 1]) * df
    return price
