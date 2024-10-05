"""Classes for technical analysis of assets."""

import math
from .utils import validate_dataframe

class StockAnalyzer:
  """Class for calculating financial metrics. """

  @validate_dataframe(columns={
    'open', 'high', 'low', 'close'})
  def __init__(self, df):
    """Create a `StockAnalyzer`
    object with OHLC data."""
    self.data = df

  @property
  def _max_periods(self):
    """Get the number of trading 
    periods in the data."""
    return self.data.shape[0]

  @property
  def close(self):
    """Get the close column of the data."""
    return self.data.close
  
  @property
  def pct_change(self):
    """Get the percent change of the close column."""
    return self.close.pct_change()

  @property
  def pivot_point(self):
    """Calculate the pivot point."""

    sum = (
      self.last_high + 
      self.last_low + 
      self.last_close
    )

    return sum / 3

  @property
  def last_close(self):
    """Get the value of the last close in the data."""
    return self.data.last('1D').close.iat[0]

  @property
  def last_high(self):
    """Get the value of the last hogh in the data."""
    return self.data.last('1D').high.iat[0]
  
  @property
  def last_low(self):
    """Get the value of the last low in the data."""
    return self.data.last('1D').low.iat[0]
  
  def alpha(self, index, return_rate):
    """Calculates the asset's alpha.

    Parameters:
    -----------
    index:
    The index to compare to.

    return_rate:
    The risk-free rate of return.

    Returns:
    --------
    Alpha, as a float.
    """
    return_rate /= 100
    r_m = self.portfolio_return(index)
    beta = self.beta(index)
    r = self.portfolio_return(self.data)
    alpha = r - return_rate - beta * (r_m - return_rate)
    return alpha
  
  def beta(self, index) -> float:
    """Calculate the beta of the asset.

    Parameters:
    -----------
    index:
    The data for the index to compare to.

    Returns:
    --------
    The calculated beta value
    """
    index_change = index.close.pct_change()
    cov = self.pct_change.cov(index_change)
    beta = cov / index_change.var()
    return beta

  def annualized_volatility(self):
    """Calculate the annualized volatility."""
    return self.daily_std() * math.sqrt(252)
      
  def corr_with(self, other):
    """Calculate the correlations 
    between dataframes.

    Parameters:
    -----------
    other:
      The other dataframe.

    Returns:
    --------
    A `pandas.Series` object
    """
    pct_change = self.data.pct_change()
    other_pct_change = other.pct_change()
    corr = pct_change.corrwith(other_pct_change)
    return corr
    
  def cumulative_returns(self):
    """Calculate cumulative returns for plotting."""
    return (1 + self.pct_change).cumprod()
    
  def cv(self):
    """Calculate the coefficient of variation for the asset.
    The lower this is, the better the risk/return tradeoff.
    """
    return self.close.std() / self.close.mean()
  
  def daily_std(self, periods=252):
    """Calculate daily standard 
    deviation of percent change.

    Parameters:
    ----------
    periods: 
      The number of periods to use for 
      the days calculation; default is 
      252 for the trading  in a year. 
      Note if you provide a number greater
      than the number of trading periods 
      in the data, `self._max_periods` will 
      be used instead.

    Returns:
    --------
    The standard deviation
    """
    c_min = min(periods, self._max_periods)
    vals = self.pct_change[-1 * c_min:]
    return vals.std()

  def is_bear_market(self):
    """Determine if a stock is in a bear market, meaning its
    return in the last 2 months is a decline of 20% or more
    """
    return self.portfolio_return(self.data.last('2M')) <= -.2

  def is_bull_market(self):
    """Determine if a stock is in a bull market, meaning its
    return in the last 2 months is an increase of >= 20%.
    """
    threshold = 0.2
    ptf_ret = self.portfolio_return(self.data.last('2M'))
    return bool(ptf_ret >= threshold)
        
  # since the method calculates the portfolio   
  # return for an index rather than the data
  # stored in self.data, it will be static
  @staticmethod
  def portfolio_return(df):
    """Calculate return assuming 
    o distribution per share.

    Parameters:
    -----------
    df:
      The asset's dataframe.

    Returns:
    --------
    The return, as a float.
    """
    start, end = df.close[0], df.close[-1]
    return (end - start) / start
  
  def qcd(self):
    """Calculate the quantile coefficient of dispersion."""

    q1, q3 = self.close.quantile([0.25, 0.75])
    qcd = (q3 - q1) / (q3 + q1)

    return qcd
  
  def resistance(self, level=1):
    """Calculate the resistance at the given level."""

    if level == 1:
      res = (2 * self.pivot_point) - self.last_low
    elif level == 2:
      res = self.pivot_point + (self.last_high - self.last_low)
    elif level == 3:
      res = self.last_high + 2 * (self.pivot_point - self.last_low)
    else:
      raise ValueError(f'Invalid resistance level: {level}')

    return res

  def sharpe_ratio(self, return_rate=0):
    """Calculates the asset's Sharpe ratio.

    Parameters:
    -----------
    return_rate:
    The risk-free rate of return.

    Returns:
    --------
    The Sharpe ratio, as a float.
    """
    cumm_rets = self.cumulative_returns()
    last_cumm_ret = cumm_rets.last('1D').iat[0]
    res = last_cumm_ret - return_rate
    std = self.cumulative_returns().std()
    return res / std    
    
  def support(self, level=1):
    """Calculate the support at the given level."""

    if level == 1:
      sup = (2 * self.pivot_point) - self.last_high
    elif level == 2:
      sup = self.pivot_point - (self.last_high - self.last_low)
    elif level == 3:
      sup = self.last_low - 2 * (self.last_high - self.pivot_point)
    else:
      raise ValueError('Not a valid level.')
    
    return sup
      
  def volatility(self, periods=252):
    """Calculate the rolling volatility.

    Parameters:
    -----------
    periods: 
      The number of periods to use for 
      the calculation; default is 252 for 
      the trading days in a year. Note if 
      you provide a number greater than 
      the number of trading periods in the
      data, `self._max_periods` will be used
      instead.

    Returns:
    --------
    A `pandas.Series` object.
    """

    periods = min(periods, self._max_periods)
    vol = self.close.rolling(periods).std() / math.sqrt(periods)

    return vol
    
class AssetGroupAnalyzer:
  """Analyzes many assets in a dataframe."""

  @validate_dataframe(columns={
    'open', 'high', 'low', 'close'})
  def __init__(self, df, group_by='name'):
    """Create an `AssetGroupAnalyzer` object with 
    a dataframe of OHLC data and column to group by.
    """

    self.data = df

    if group_by not in self.data.columns:
      raise ValueError(
        f'The `group_by` column f"{group_by}" '
        'not found in the dataframe!'
      )
    
    self.group_by = group_by
    self.analyzers = self._composition_handler()

  def _composition_handler(self):
    """Create a dictionary mapping each group to its analyzer,
    taking advantage of composition instead of inheritance.
    """
    return {
      group: StockAnalyzer(data)
      for group, data in self.data.groupby(self.group_by)
    }
  
  def analyze(self, func_name, **kwargs):
    """Run a `StockAnalyzer` method on all assets.

    Parameters:
    -----------
    func_name:
    The name of the method to run.

    kwargs:
    Additional arguments to pass down.

    Returns:
    --------
    A dictionary mapping each asset to the result
    of the calculation of that function.
    """

    if not hasattr(StockAnalyzer, func_name):
      raise ValueError(
        f'StockAnalyzer has no "{func_name}" method!'
      )
    
    if not kwargs:
      kwargs = {}

    return {
      group: getattr(analyzer, func_name)(**kwargs)
      for group, analyzer in self.analyzers.items()
    }
  