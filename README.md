# Stock_predictor<br>
A Python-based Monte Carlo simulation tool for predicting future stock price distributions using Geometric Brownian Motion (GBM) and historical market data.<br>
<br>
__Overview__<br>
This application uses Monte Carlo simulation to model potential future stock price paths based on historical volatility and returns. By running thousands of simulations, it generates a probabilistic distribution of possible outcomes, helping investors understand risk and potential returns.

__What it Does__<br>
Automatic data fetching from Yahoo Finance using yfinance <br>
Use Geometric Brownian Motion (GBM) modeling<br>
Price path simulations showing all possible trajectories<br>
Probability distributions of final prices<br>
Cauculate the mean and median final prices<br>
5th and 95th percentile outcomes (confidence intervals)<br>
<br>
__Geometric Brownian Motion (GBM)__<br>
dS = S * (μ * dt + σ * √dt * Z)<br>
<br>
Where:<br>
S = Current stock price<br>
μ (mu) = Expected return (drift)<br>
σ (sigma) = Volatility (standard deviation)<br>
dt = Time step<br>
Z = Random normal variable<br>
<br>
__Disclaimer__<br>
This tool was created as a personal project for studying and does not constitute financial advice, investment recommendations, or trading signals.<br>
This does not model the Black Swan Effect or any Company specific event that can affects the outcome.


