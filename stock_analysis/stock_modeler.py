"""Module with classes for modeling assets."""

import utils

class StockModeler:
  """Class for modeling stocks. """

  def arima(self, df):
    pass
  
  def arima_predictions(self, df, arima_model_fitted, start, end, plot):
    pass
  
  def decompose(self, df, period, model):
    pass
  
  def plot_residuals(self, model_fitted, freq):
    pass
  
  def regression(self, df):
    pass
  
  def regression_predictions(self, df, model, start, end, plot):
    pass