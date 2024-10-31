"""Analyzing Facebook stock prices from 2018."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plots import show_plot, plot_regression_with_residues

# read data from the file
data = pd.read_csv(
  'data/facebook/stock_prices_2018.csv',
  index_col='date',
  parse_dates=True
)

# prep data for analysis
data = data.sort_index() # sort the data by date
data = data.assign(
  volume_log = np.log(data.volume),
  daily_diff = data.high - data.low,
  quarter = data.index.quarter,
  moving_average=lambda x: x.close.rolling(window=20).mean()
)

# analyze the trends in the stock prices over time
axes = data[['open', 'high', 'low', 'close']].plot(
  subplots=True,
  layout=(2, 2),
  figsize=(12, 5)
)

plt.xlabel('Date')
for ax in axes.flatten():
  ax.set_ylabel('Price (USD)')
show_plot(suptitle='Facebook 2018 Stock Data') # give the title to the entire figure


# use Facebook stock's closing price and the 20-day moving average
# to get insights into the stock's performance over time
data.plot(
  y=['close', 'moving_average'],
  title='Facebook Closing Price in 2018',
  label=['closing_price', ' 20D moving average'],
  style=['-', '--']
)
plt.legend(loc='lower left')
plt.ylabel('Price (USD)')
show_plot()

breakpoint()
pass

# calculate the correlation matrix
# using Pearson correlation coefficient
correlations = data.corr()

# display the correlation between the OHLC stock prices,
# the log of volume traded, and daily difference between 
# the highest and lowest prices in form of a heatmap
sns.heatmap(
  correlations, 
  annot=True, 
  center=0, # put values of 0 (no correlation) at the center of the heatmap
  vmin=-1, 
  vmax=1
)
show_plot('Correlation between OHLC stock prices, volume, and daily difference')


# investigane the correlations between the columns in 
# the Facebook data as scatter plots instead of the heatmap
sns.pairplot(
  data[['open', 'high', 'low', 'close', 'volume', 'quarter']],
  diag_kind='kde', # use kernel density estimate for the diagonal plots
  hue='quarter', # see how the facebook performed on a quarterly basis
  # kind='reg' # nice option but it makes the plot less readable
)
show_plot('Correlation between OHLC stock prices, volume, and daily difference')

# observe max absolute change in the stock 
# prices and the corresponding volume traded
sns.jointplot(
  x='volume_log',
  y='daily_diff',
  kind='hex', # the hex shapes may be beneficial if the dots overlap
  data=data[['volume_log', 'daily_diff']]
)
show_plot('Relationship between the volume traded and daily price change')

# visualize the data using contour plot
sns.jointplot(
  x='volume_log',
  y='daily_diff',
  kind='kde',
  data=data[['volume_log', 'daily_diff']]
)
show_plot('Relationship between the volume traded and daily price change')

# display the linear regression line through the scatter plot, 
# along with a confidence band surrounding the line
sns.jointplot(
  x='volume_log',
  y='daily_diff',
  kind='reg',
  data=data[['volume_log', 'daily_diff']]
)
show_plot('Relationship between the volume traded and daily price change')

# though the relationshoip appears to be linear,
# after checking the residuals we can see that the
# residuals appear to be greateer at higher quantities
# of volumes traded, indicating that a different model 
# needs to be used to model this relationship
sns.jointplot(
  x='volume_log',
  y='daily_diff',
  kind='resid',
  data=data[['volume_log', 'daily_diff']]
)
show_plot('Residuals')

# analyze the logfarithm of volume traded and the daily difference
# between the highest and lowest prices in the stock
plot_regression_with_residues(data[['volume_log', 'daily_diff']])
show_plot('Linear regression and residuals')

# plot regressions between stock volumnes 
# and daily differences across quarters of data
sns.lmplot(
  data,
  x='volume_log',
  y='daily_diff',
  col='quarter'
)
show_plot('Linear regression and residuals')

