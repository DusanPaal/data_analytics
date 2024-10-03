"""Gather select stock data."""

# standard library imports
import datetime as dt
import re

# third-party library imports
import pandas as pd
import pandas_datareader.data as web

# local library imports
from .utils import label_sanitizer

class StockReader:
    """Class for reading financial data from websites."""

    _index_tickers = {
      'S&P 500': '^GSPC',
      'Dow Jones': '^DJI',
      'NASDAQ': '^IXIC'
    }

    def __init__(self, start, end=None) -> None:
      """Create a `StockReader` object for 
      reading across a given date range.

      Parameters:
      -----------
      start: 
        The first date to include, as a datetime
        object or a string in the format 'YYYYMMDD'.

      end: 
        The last date to include, as a datetime
        object or string in the format 'YYYYMMDD'.
        Defaults to today if not provided.
      """

      # parse the date rage
      # TODO: add date format validation
      self.start, self.end = map(
        lambda x: x.strftime('%Y%m%d') if isinstance(x, dt.date) else re.sub(r'\D', '', start, end),
        [start, end or dt.date.today()]
      )

      # validate the date range
      if self.start > self.end:
        raise ValueError('Start date must be before end date!') 

    def get_bitcoin_data(self, currency_code):
      """
      Get bitcoin historical OHLC data for given date range.
      Parameter:
      - currency_code: The currency to collect the bitcoin
      data in, e.g. USD or GBP.
      Returns:
      A `pandas.DataFrame` object with the bitcoin data.
      """
      data = self.get_ticker_data(f'BTC-{currency_code}')
      return data.loc[self.start:self.end] # clip the data

    @label_sanitizer
    def get_forex_rates(self, from_currency, to_currency, **kwargs):
      """Retrieve daily foreign exchange rates data.
      
      """
      pass
    
    def get_index_ticker(cls, index: str) -> str | None:
      """Look up the ticker (stock market symbol)
      for a specific index, if known.

      Parameters:
      -----------
      index: 
        The name of the index. Check `available_tickers` 
        for full list which includes:
          - 'S&P 500' for S&P 500,
          - 'Dow Jones' for Dow Jones Industrial Average,
          - 'NASDAQ' for NASDAQ Composite Index

      Returns:
      --------
      The ticker as a `str` if known, otherwise `None`.

      Example:
      --------
      >>> StockReader.get_index_ticker('S&P 500')
      >>> '^GSPC'
      """

      if not isinstance(index, str):
        raise TypeError('Index must be a string!')

      index = index.strip().upper()

      return cls._index_tickers.get(index, None)

    def get_risk_free_rate_of_return(self, last: bool = True):
      """Collect the risk-free rate of return.

      This method retrieves the daily rate for a 10-year US
      T-bill from FRED (https://fred.stlouisfed.org/series/DGS10)

      Parameters:
      -----------
      last:
        If `True`, return the rate on the last
        date in the date range else, return a `Series`
        object for the rate each day in the date range.

      Returns:
      --------
      A single value or a `pandas.Series` object.
      """

      df = web.DataReader('DGS10', 'fred', self.start, self.end)
      df.index.rename('date', inplace=True)
      data = data.squeeze()

      if last and isinstance(data, pd.Series):
        return data.asof(self.end)
      
      return data
    
    @label_sanitizer
    def get_ticker_data(self, ticker) -> pd.DataFrame | pd.Series:
      """Get historical OHLC data for given date range and ticker.

      Parameter:
      ----------
      ticker:
        The stock symbol to lookup as a string.

      Returns:
      --------
      A `pandas.DataFrame` object with the stock data.
      """
      return web.get_data_yahoo(ticker, self.start, self.end)

    def get_index_data(self, index: str):
      """Retreive data for an index on the stock market.
      
      Get historical OHLC data from Yahoo! Finance
      for the chosen index for given date range.

      Parameters:
      -----------
      index: 
        String representing the index of the data to fetch:
        - 'S&P 500' for S&P 500,
        - 'Dow Jones' for Dow Jones Industrial Average,
        - 'NASDAQ' for NASDAQ Composite Index

      Returns:
      --------
      A `pandas.DataFrame` object with the index data.
      """

      if index not in self.available_tickers:
        raise ValueError(
          'Index not supported. Available tickers'
          f"are: {', '.join(self.available_tickers)}"
        )

      return self.get_ticker_data(self.get_index_ticker(index))

    @property
    def available_tickers(self):
      """Return a list of available index tickers."""
      return list(self._index_tickers.keys())
