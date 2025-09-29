# Stock_predictor
A Python-based Monte Carlo simulation tool for predicting future stock price distributions using Geometric Brownian Motion (GBM) and historical market data.

__Overview__
This application uses Monte Carlo simulation to model potential future stock price paths based on historical volatility and returns. By running thousands of simulations, it generates a probabilistic distribution of possible outcomes, helping investors understand risk and potential returns.

__What it Does__
Automatic data fetching from Yahoo Finance using yfinance
Use Geometric Brownian Motion (GBM) modeling
Price path simulations showing all possible trajectories
Probability distributions of final prices
Cauculate the mean and median final prices
5th and 95th percentile outcomes (confidence intervals)

__Geometric Brownian Motion (GBM)__
dS = S * (μ * dt + σ * √dt * Z)

Where:
S = Current stock price
μ (mu) = Expected return (drift)
σ (sigma) = Volatility (standard deviation)
dt = Time step
Z = Random normal variable

__Disclaimer__
This tool was created as a personal project for studying and does not constitute financial advice, investment recommendations, or trading signals.
This does not model the Black Swan Effect or any Company specific event that can affects the outcome.


