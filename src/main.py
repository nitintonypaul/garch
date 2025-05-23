#Importing dependencies
import yfinance as yf
import numpy as np
import garch_est as ge #c++ module

#Obtaining stock from the user
stock = input("Enter a stock: ")
ticker_symbol = yf.Ticker(stock)

#Collecting data from the past 395 days and using the first 30 days to obtain volatility before 365 days
data = ticker_symbol.history(period="395d")
prices = data["Close"][0:30]

#Computing shock array - used to compute and use shock value
returns_array = data["Close"][30:]

#Computing log returns
log_returns = np.log(prices / prices.shift(1)).dropna()

#Computing average log returns (mu) and volatility 
mu = np.mean(log_returns)
vol = log_returns.std()

#Converting returns array into shocks
shock_arr = list(np.log(returns_array / returns_array.shift(1)).dropna() - mu)

#Finding volatility using cpp module
expected_vol = ge.estimate_vol(len(shock_arr), vol, shock_arr)

#Displaying result
print("==============================================================")
print(f"Stock chosen for analysis: {stock}")
print(f"Volatility before 1 year from today: {vol*100}%")
print(f"Volatility predicted for tomorrow: {expected_vol*100}%")
print("==============================================================")