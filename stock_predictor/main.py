import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === User settings ===
ticker = "AAPL"  # Change this to any stock symbol
period = "1y"  # 1 year of historical data
M = 100  # number of simulations
T = 1  # time horizon in years
dt = 1 / 252  # daily steps
N = int(T / dt)  # number of steps

print(f"Running simulation with {N} steps over {T} year(s)")

# === Fetch historical data ===
try:
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    # Handle both single-level and multi-level column structure
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data["Close"].iloc[:, 0]  # Take first column if multi-level
    else:
        close_prices = data["Close"]

except Exception as e:
    print(f"Error fetching data: {e}")
    # Use sample data as fallback
    print("Using sample data...")
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    close_prices = pd.Series(150 + np.cumsum(np.random.randn(252) * 2), index=dates)

# === Calculate returns ===
returns = np.log(close_prices / close_prices.shift(1)).dropna()

# === Parameters ===
S0 = float(close_prices.iloc[-1])  # last closing price (ensure it's a scalar)
mu = float(returns.mean() * 252)  # annualized mean return
sigma = float(returns.std() * np.sqrt(252))  # annualized volatility

print(f"Stock: {ticker}")
print(f"Current Price (S0): ${S0:.2f}")
print(f"Expected Annual Return (mu): {mu:.2%}")
print(f"Annual Volatility (sigma): {sigma:.2%}")

# === Monte Carlo Simulation (Geometric Brownian Motion) ===
print("Running Monte Carlo simulation...")

# Initialize the simulation array
simulations = np.zeros((N, M))

# Set the initial price for all simulations
simulations[0, :] = S0

# Generate all random numbers at once for efficiency
random_shocks = np.random.normal(0, 1, (N - 1, M))

# Simulate each path
for j in range(M):
    for i in range(1, N):
        # Geometric Brownian Motion formula: dS = S * (mu*dt + sigma*sqrt(dt)*Z)
        drift = mu * dt
        diffusion = sigma * np.sqrt(dt) * random_shocks[i - 1, j]

        # Update price using GBM
        simulations[i, j] = simulations[i - 1, j] * np.exp(drift + diffusion)

print("Simulation completed!")

# === Calculate statistics ===
final_prices = simulations[-1, :]
mean_final_price = np.mean(final_prices)
median_final_price = np.median(final_prices)
percentile_5 = np.percentile(final_prices, 5)
percentile_95 = np.percentile(final_prices, 95)

print(f"\n=== Results after {T} year(s) ===")
print(f"Mean final price: ${mean_final_price:.2f}")
print(f"Median final price: ${median_final_price:.2f}")
print(f"5th percentile: ${percentile_5:.2f}")
print(f"95th percentile: ${percentile_95:.2f}")
print(f"Probability of profit: {np.sum(final_prices > S0) / M:.1%}")

# === Plot results ===
plt.figure(figsize=(12, 8))

# Plot 1: Price paths
plt.subplot(2, 1, 1)
time_axis = np.linspace(0, T * 252, N)  # Convert to trading days for x-axis
plt.plot(time_axis, simulations, lw=0.7, alpha=0.6)
plt.axhline(y=S0, color='red', linestyle='--', linewidth=2, label=f'Current Price: ${S0:.2f}')
plt.title(f'Monte Carlo Simulation: {M} Price Paths for {ticker}')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Distribution of final prices
plt.subplot(2, 1, 2)
plt.hist(final_prices, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(S0, color='red', linestyle='--', linewidth=2, label=f'Current Price: ${S0:.2f}')
plt.axvline(mean_final_price, color='green', linestyle='-', linewidth=2, label=f'Mean: ${mean_final_price:.2f}')
plt.axvline(percentile_5, color='orange', linestyle=':', linewidth=2, label=f'5th %ile: ${percentile_5:.2f}')
plt.axvline(percentile_95, color='orange', linestyle=':', linewidth=2, label=f'95th %ile: ${percentile_95:.2f}')

plt.title(f'Distribution of {ticker} Stock Price after {T} Year(s)')
plt.xlabel('Stock Price ($)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Risk metrics ===
returns_simulation = (final_prices - S0) / S0
var_95 = np.percentile(returns_simulation, 5)  # Value at Risk (95% confidence)
cvar_95 = np.mean(returns_simulation[returns_simulation <= var_95])  # Conditional VaR

print(f"\n=== Risk Metrics ===")
print(f"Value at Risk (95%): {var_95:.2%}")
print(f"Conditional Value at Risk (95%): {cvar_95:.2%}")
print(f"Maximum simulated loss: {np.min(returns_simulation):.2%}")
print(f"Maximum simulated gain: {np.max(returns_simulation):.2%}")