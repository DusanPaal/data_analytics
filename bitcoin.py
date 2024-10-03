import pandas as pd
import stock_analysis

from stock_analysis.utils import group_stocks, describe_group, make_portfolio

# read data from file
data = pd.read_csv(
  r'data/bitcoin/bitcoin.csv',
  index_col='date',
  parse_dates=True
)

reader = stock_analysis.StockReader('2019-01-01', '2020-12-31')

fb, apple, amazon, netflix, google = (
  reader.get_ticker_data(ticker)
  for ticker in ['FB', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']
)

sp = reader.get_index_data('S&P 500')
bitcoin = reader.get_bitcoin_data('USD')

# what does involve story telling in data analysis?

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

portfolio = make_portfolio(all_assets)
print(portfolio)

breakpoint()
pass