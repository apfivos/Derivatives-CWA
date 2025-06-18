# 📈 Advanced Option Pricing 

This repository contains my contribution to the group coursework in the **Derivative Securities** module in the MSc Quantitative Finance at the University of Manchester.

The objective of the assignment was to price a variety of financial derivatives written on a specific underlying stock, using a wide range of analytical and numerical methods taught throughout the course.

## ✍️ Scope of Contribution

This repo includes my implementation for the following tasks:

- **Task 1:** Black-Scholes pricing and verification of put-call parity
- **Task 2:** Binomial tree model for European put option pricing
- **Task 3:** Pricing of an American floating-strike lookback put option using a binomial tree
- **Task 4:** Monte Carlo Simulation Methods:
  - 4a: European put option with variance reduction via Halton sequences
  - 4b: Monte Carlo with stochastic interest rate paths
  - 4c: Down-and-out put (barrier option)
- **Task 7:** American put option pricing using the Longstaff-Schwartz regression-based algorithm

These implementations were developed and validated as part of a collaborative group project, with each team member focusing on a subset of tasks.

---




### ▶️ Run the Code

Install dependencies using:
```bash
pip install -r requirements.txt
```

```bash
python main.py
```

---



## 📚 Notes
- The stock price used throughout the project is fixed and predefined for consistency across tasks.
- Halton sequences were employed in Monte Carlo simulation for improved convergence.
- All code is modular and ready for further experimentation and extension.

---

## ©️ Author
