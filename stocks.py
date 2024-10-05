import pandas as pd
import stock_analysis
import sys
import matplotlib.pyplot as plt
from stock_analysis.utils import (
  group_stocks, describe_group, make_portfolio
)

from stock_analysis import (
  StockModeler, AssetGroupAnalyzer,
  StockVisualizer, AssetGroupVisualizer,
  StockReader, DataMissingError
)

reader = StockReader('2019-01-01', '2020-12-31')

try:
  fb, apple, amazon, netflix, google = (
    reader.get_ticker_data(ticker)
    for ticker in ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']
    # Facebook was renamed to Meta
  )
except DataMissingError as err:
  print(err)
  sys.exit(0)

# check the evolution of the closing price of Netflix using ARIMA
decomposition = StockModeler.decompose(netflix, 20) # seasonal
fig = decomposition.plot()
fig.suptitle('Netflix Stock Price Time Series Decomposition', y=1)
fig.set_figheight(6)
fig.set_figwidth(10)
fig.tight_layout()
plt.show()

# autocorrelation_plot() function from the pandas.plotting 
# module will help find a good value for ar, for now we use 
# the vals for faster calculation
arima_model = StockModeler.arima(netflix, ar=10, i=1, ma=5)
print(arima_model.summary())

# The residuals should have a mean of 0 and equal variance 
# throughout, meaning that they do not depend on the independent 
# variable - the date, in this case.
StockModeler.plot_residuals(arima_model)
plt.show()
# the residuals are heteroschedastic - centered around 0
# but their variance tends to increase over time

# lets look at the same data but with a linear regression model
x, y, lm = StockModeler.regression(netflix)
print(lm.summary())
StockModeler.plot_residuals(lm)
plt.show()

# now compare the predictions of the ARIMA 
# and regression models to the actual data
start, end = '2021-01-01', '2021-01-14'
january = stock_analysis.StockReader(start, end)
jan_netflix = january.get_ticker_data('NFLX')
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
arima_ax = StockModeler.arima_predictions(
  netflix, arima_model, start=start, end=end,
  ax=axes[0], title='ARIMA', color='b'
)
jan_netflix.close.plot(
  ax=arima_ax, style='b--', label='actual close'
)
arima_ax.legend()
arima_ax.set_ylabel('price ($)')
linear_reg = StockModeler.regression_predictions(
  netflix, lm, start=start, end=end,
  ax=axes[1], title='Linear Regression', color='b'
)
jan_netflix.close.plot(
  ax=linear_reg, style='b--', label='actual close'
)
linear_reg.legend()
linear_reg.set_ylabel('price ($)')
plt.show()

# read in the S&P 500 data and Bitcoin data
sp = reader.get_index_data('S&P 500')
bitcoin = reader.get_bitcoin_data('USD')

faang = group_stocks({
  'Facebook': fb, 'Apple': apple,
  'Amazon': amazon, 'Netflix': netflix,
  'Google': google
})

faang_sp = group_stocks({
  'Facebook': fb, 'Apple': apple, 'Amazon': amazon,
  'Netflix': netflix, 'Google': google, 'S&P 500': sp
})

all_assets = group_stocks({
  'Bitcoin': bitcoin, 'S&P 500': sp, 'Facebook': fb,
  'Apple': apple, 'Amazon': amazon, 'Netflix': netflix,
  'Google': google
})

desc = describe_group(all_assets)
print(desc)
all_assets_analyzer = AssetGroupAnalyzer(all_assets)
all_assets_analyzer.analyze('cv')
all_assets_analyzer.analyze('annualized_volatility')
all_assets_analyzer.analyze('is_bull_market')

rf = reader.get_risk_free_rate_of_return()
all_assets_analyzer.analyze('alpha', index=sp, return_rate=rf)
cumm_returns = all_assets_analyzer.analyze('cumulative_returns')
stock_analysis.stock_visualizer.get_cycler(cumm_returns)

portfolio = make_portfolio(all_assets)
print(portfolio)

netflix_viz = StockVisualizer(netflix)

# plot the closing price of Netflix and shade the region
# between the 30-day and 90-day moving averages
shader = netflix_viz.region_shader(
  x=('2019-10-01', '2020-07-01'),
  color='blue', alpha=0.1
)

netflix_viz.moving_average(
  title='Netflix Closing Price',
  column='close', 
  periods=['30D', '90D'],
  ylabel='price ($)',
  shader=shader
)

# plot the closing price of Netflix and shade the region
# between the 30-day and 90-day exponential moving averages
shader = netflix_viz.region_shader(
  x=('2020-04-01', '2020-10-01'),
  color='blue', alpha=0.1
)

netflix_viz.exp_smoothing(
  title='Netflix Closing Price',
  column='close',
  periods=[30, 90], 
  ylabel='price ($)'
)

# plot the copen price of Netflix in terms of after hours trading
netflix_viz.after_hours_trades()

# resample the data into 2-week intervals to 
# improve the visibility of the candlesticks:
netflix_viz.candlestick(
  resample='2W', 
  volume=True, 
  xrotation=90, 
  datetime_format='%Y-%b -'
)

# reset the plotting style before
# creating another visualization
netflix_viz.reset_plottig() 

# Comparing Netflix to the S&P 500
netflix_viz.jointplot(sp, 'close')

# visualize the correlations
# between Netflix and Amazon
netflix_viz.correlation_heatmap(amazon)

# compare Netflix to Tesla here to 
# see one stock surpassing another
tesla = reader.get_ticker_data('TSLA')
change_date = (tesla.close > netflix.close).idxmax()
ax = netflix_viz.fill_between_other(tesla)

netflix_viz.add_reference_line(
  ax, x=change_date, color='k', linestyle=':', alpha=0.5,
  label=f'TSLA > NFLX {change_date:%Y-%m-%d}'
)

all_assets_viz = AssetGroupVisualizer(all_assets)
all_assets_viz.heatmap()

faang_sp_viz = AssetGroupVisualizer(faang_sp)
bitcoin_viz = StockVisualizer(bitcoin)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

faang_sp_viz.evolution_over_time(
  'close', ax=axes[0], style=faang_sp_viz.group_by
)

bitcoin_viz.evolution_over_time(
  'close', ax=axes[1], label='Bitcoin'
)

plt.show()

breakpoint()
pass
