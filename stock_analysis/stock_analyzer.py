"""Module with classes for calculating financial metrics."""

import utils

class StockAnalyzer:
  """Class for calculating financial metrics. """

  def __init__(self):
    """Initialize the StockAnalyzer object."""

    self.data = None
    self.close = None
    self.last_close = None
    self.last_high = None
    self.last_low = None
    self.pct_change = None
    self.pivot_point = None

  def alpha(self, index, r_f):
    pass
  
  def annualized_volatility(self):
    pass
  
  def beta(self, index):
    pass
  
  def corr_with(self, other):
    pass
  
  def cummulative_returns(self, df):
    pass
  
  def cv(self):
    pass
  
  def daily_std(self, periods):
    pass
  
  def is_bear_market(self):
    pass
  
  def is_bull_market(self):
    pass
  
  def portfolio_return(self, df):
    pass
  
  def qcd(self):
    pass
  
  def resistance(self, level):
    pass
  
  def shape_ratio(self, r_f):
    pass
  
  def support(self, level):
    pass
  
  def volatility(self, periods):
    pass
  
class AssetGroupAnalyzer:
  """..."""

  def __init__(self):
    self.analyzers = []
    self.group_by = ''
    self.data = None

  def analyze(self, func_name):
    pass
