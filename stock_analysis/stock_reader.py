"""Gather select stock data."""

# standard library imports
import datetime as dt
import re
from typing import Optional

# third-party library imports
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

# local library imports
from .utils import label_sanitizer

class DataMissingError(Exception):
  """Custom exception for missing 
  data in the remote data source.
  """

class StockReader:
    """Class for reading financial data from websites."""

    _index_tickers = {
        'S&P 500': '^GSPC', 'Dow Jones': '^DJI', 'NASDAQ': '^IXIC', # US
        'S&P/TSX Composite Index': '^GSPTSE', # Canada
        'IPC Mexico': '^MXX', # Mexico
        'IBOVESPA': '^BVSP', # Brazil
        'Euro Stoxx 50': '^STOXX50E', # Europe
        'FTSE 100': '^FTSE', # UK
        'CAC 40': '^FCHI', # France
        'DAX': '^GDAXI', # Germany
        'IBEX 35': '^IBEX', # Spain
        'FTSE MIB': 'FTSEMIB.MI', # Italy
        'OMX Stockholm 30': '^OMX', # Sweden
        'Swiss Market Index': '^SSMI', # Switzerland
        'Nikkei': '^N225', # Japan
        'Hang Seng': '^HSI', # Hong Kong
        'CSI 300': '000300.SS', # China
        'S&P BSE SENSEX': '^BSESN', # India
        'S&P/ASX 200': '^AXJO', # Australia
        'MOEX Russia Index': '^IMOEX.ME' # Russia
    } # to add more, consult https://finance.yahoo.com/world-indices/

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
        lambda x: x.strftime('%Y%m%d') if isinstance(x, dt.date) else re.sub(r'\D', '', x),
        [start, end or dt.date.today()]
      )

      # validate the date range
      if self.start > self.end:
        raise ValueError('Start date must be before end date!') 

    def get_bitcoin_data(self, currency_code: str) -> pd.DataFrame:
      """Get bitcoin historical OHLC data for given date range.

      Parameters:
      -----------
      currency_code:
        The currency to collect the bitcoin
        data in, e.g. USD or GBP.

      Returns:
      --------
      A `pandas.DataFrame` object with the bitcoin data.
      """
      data = self.get_ticker_data(f'BTC-{currency_code}')
      return data.loc[self.start:self.end] # clip the data

    @label_sanitizer
    def get_forex_rates(self, from_currency: str, to_currency: str, **kwargs) -> pd.DataFrame:
        """Get daily foreign exchange rates from AlphaVantage.

        Note:
        -----
        This requires an API key, which can be obtained for free at
        https://www.alphavantage.co/support/#api-key. To use this method, 
        you must either store it as an environment variable called 
        `ALPHAVANTAGE_API_KEY` or pass it in to this method as `api_key`.

        Parameters:
        -----------
        from_currency:
          The currency you want the exchange rates for.

        to_currency:
          The target currency.

        Returns:
        --------
        A `pandas.DataFrame` with daily exchange rates.
        """

        data = web.DataReader(
            f'{from_currency}/{to_currency}', 'av-forex-daily',
            start=self.start, end=self.end, **kwargs
        ).rename(pd.to_datetime)

        data.index.rename('date', inplace=True)

        return data
    
    @classmethod
    def get_index_ticker(cls, index: str) -> Optional[str]:
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
      """Get the risk-free rate of return using the 10-year US Treasury bill.
      Source: FRED (https://fred.stlouisfed.org/series/DGS10)

      Parameters:
      -----------
      last:
        If `True` (default), return the rate on the last
        date in the date range else, return a `Series`
        object for the rate each day in the date range.

      Returns:
      --------
      A single value or a `pandas.Series` object.
      """

      data = web.DataReader('DGS10', 'fred', self.start, self.end)
      data.index.rename('date', inplace=True)
      data = data.squeeze()

      if last and isinstance(data, pd.Series):
        return data.asof(self.end)
      
      return data
    
    @label_sanitizer
    def get_ticker_data(self, ticker) -> pd.DataFrame:
      """Get historical OHLC data for given date range and ticker.

      Parameter:
      ----------
      ticker:
        The stock symbol to lookup as a string.

      Returns:
      --------
      A `pandas.DataFrame` object with the stock data.

      Raises:
      -------
      DataMissingError:
      If no data is found for the ticker.
      """ 

      try:
          data = web.get_data_yahoo(ticker, self.start, self.end)
      except Exception as exc: 
          # API issue upstream – switch to yfinance
          # https://github.com/pydata/pandas-datareader/issues/952
          print(f'Error fetching finance data from Yahoo for ticker {ticker}!: {exc}')
          print('Handling the error by attempting to fetch the data using yfinance...', end='')
          start, end = (
              dt.datetime.strptime(str_date, '%Y%m%d')
              for str_date in (self.start, self.end)
          )
          data = yf.download(
              ticker, start, end + dt.timedelta(days=1),
              progress=False, ignore_tz=True
          )

          if data.empty:
            raise DataMissingError(
              f'No data found for ticker "{ticker}" using '
              'yfinance! The ticker may have been delisted.')

          print('success!')

      return data

    def get_index_data(self, index: str):
      """Get historical OHLC data from Yahoo! Finance
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
