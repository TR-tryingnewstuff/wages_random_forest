#%%
import numpy as np
from full_fred.fred import Fred
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as xp

fred = Fred('/home/fast-pc-2023/Téléchargements/python/master_rl/api_key.txt')
risk_free_rate = float(fred.get_series_df('DTB4WK')['value'].values[-1]) / 100
msft = yf.Ticker('MSFT').history()
std = msft['Close'].std() / 100
price = msft['Close'].iloc[-1]

print('The underlying price is : {} \nThe underlying volatility is : {} \nThe Risk free rate is : {}'.format(price, std, risk_free_rate))

def price_lookback_put_option(r, sigma, T, n, N, P0):
    """
    Prices a lookback put option using a Monte Carlo simulation.

    :param r: The risk-free interest rate
    :param sigma: The volatility of the underlying asset
    :param T: The time to expiration of the option
    :param n: The number of time steps in the simulation
    :param N: The number of simulation paths
    :param P0: The initial price of the underlying asset
    :return: A tuple containing the estimated price and the 95% confidence interval
    """
    dt = T / n  # time step size
    
    # Simulate N paths for the underlying asset price
    paths = np.zeros((N, n + 1))
    paths[:, 0] = P0
    for i in range(1, n + 1):
        z = np.random.standard_normal(N)  # draws from standard normal distribution
        paths[:, i] = paths[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # Calculate the maximum price for each path
    max_prices = np.max(paths, axis=1)

    # Calculate the discounted payoff for each path
    payoffs = np.exp(-r * T) * np.maximum(max_prices - paths[:, -1], 0)

    # Estimate the option price
    option_price = np.mean(payoffs)

    # Calculate the confidence interval if required
    std_error = np.std(payoffs) / np.sqrt(N)
    confidence_interval = (option_price - 1.96 * std_error, option_price + 1.96 * std_error)

    return option_price, confidence_interval

# Example usage:
r = float(risk_free_rate) / 100
sigma = std
T = 1.0
n = 365
N = 100
P0 = price

price, conf_int = price_lookback_put_option(r, sigma, T, n, N, P0)
print(f"Estimated Lookback Put Option Price: {price}")
print(f"95% Confidence Interval: {conf_int}")

dt = T / n 
paths = np.zeros((N, n + 1))
paths[:, 0] = P0
for i in range(1, n + 1):
    z = np.random.standard_normal(N)  # draws from standard normal distribution
    paths[:, i] = paths[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    
plt.plot(pd.DataFrame(paths.T, index=pd.date_range('2023-11-01', periods=n + 1)))
plt.title('Paths of 100 simulations')
plt.ylabel('Underlying price')

#%%
plt.hist(paths[:, -1], bins=50, density=True, range=(170, 800))
plt.title('Distribution function for 100 000 simulations')
plt.xlabel('Price of the underlying at time T')
plt.ylabel('Probability of occurence')
#%%
plt.hist(paths.max(axis=1), bins=35, range=(375, 700), density=True)
plt.title('Distribution function for 100 000 simulations')
plt.xlabel('Maximum Price over the period [0, T]')
plt.ylabel('Probability of occurence')
# %%

# %%

def price_butterfly_call_option(r, sigma, T, n, N, K1, K2, K3, P0):
    """
    Prices a Butterfly call option using a Monte Carlo simulation.
    :param r: The risk-free interest rate
    :param sigma: The volatility of the underlying asset
    :param T: The time to expiration of the option
    :param n: The number of time steps in the simulation
    :param N: The number of simulation paths
    :param K1: The lower strike price
    :param K2: The middle strike price
    :param K3: The higher strike price
    :param P0: The initial price of the underlying asset
    :return: A tuple containing the estimated price and the 95% confidence interval
    """
    dt = T / n # time step size
    # Simulate N paths for the underlying asset price
    paths = np.zeros((N, n + 1))
    paths[:, 0] = P0
    for i in range(1, n + 1):
        z = np.random.standard_normal(N) # draws from standard normal distribution
        paths[:, i] = paths[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    # Calculate the payoff for each path for a Butterfly call option
    payoffs = np.exp(-r * T) * (np.maximum(paths[:, -1] - K1, 0) - 2 * np.maximum(paths[:, -1] - K2, 0) +np.maximum(paths[:, -1] - K3, 0) )
    # Estimate the option price
    option_price = np.mean(payoffs)
    # Calculate the confidence interval if required
    std_error = np.std(payoffs) / np.sqrt(N)
    confidence_interval = (option_price - 1.96 * std_error, option_price + 1.96 * std_error)
    
    return option_price, confidence_interval

# Example usage:

T = 1.0
n = 100
N = 10000
K1 = price - 10
K2 = price
K3 = price + 10
P0 = price
price, conf_int = price_butterfly_call_option(r, sigma, T, n, N, K1, K2, K3, P0)
print(f"Estimated Butterfly Call Option Price: {price}")
print(f"95% Confidence Interval: {conf_int}")

# %%