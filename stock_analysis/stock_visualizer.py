"""Module with classes for visualizing assets."""

import utils
from abc import ABC, abstractmethod

class Visualizer(ABC):
  """Class for visualizing stocks. """

  @abstractmethod
  def add_reference_line(self, ax, x, y):
    pass
  
  @abstractmethod
  def after_hours_trades(self):
    pass

  @abstractmethod
  def boxplot(self):
    pass

  @abstractmethod
  def evolution_over_time(self, column):
    pass  

  @abstractmethod
  def exp_smoothing(self, column, periods):
    pass

  @abstractmethod
  def histogram(self, column):
    pass

  @abstractmethod
  def moving_average(self, column, periods):
    pass

  @abstractmethod
  def pairplot(self):
    pass

  @abstractmethod
  def shade_region(self, ax, x, y):
    pass

class StockVisualizer(Visualizer):
  """visualize a single asset."""

  def after_hours_trades(self):
    pass

  def boxplot(self):
    pass

  def candlestick(self, date, range, resample, volume):
    pass

  def correlation_heatmap(self, other):
    pass

  def evolution_over_time(self, column):
    return super().evolution_over_time(column)
  
  def fill_between(self, y1, y2, title, label_higher, label_lower, figsize, legend_x):
    pass

  def fill_between_other(self, otehr_df, figsize):
    pass

  def histogram(self, column):
    pass

  def jointplot(self, other, column):
    pass

  def open_to_close(self, figsize):
    pass

  def pairplot(self):
    pass


class AssetGroupVisualizer(Visualizer):
  """visualize multiple assets"""

  def __init__(self):
    self.group_by = ''

  def after_hours_trades(self):
    pass

  def boxplot(self):
    pass

  def evolution_over_time(self, column):
    return super().evolution_over_time(column)
  
  def heatmap(self):
    pass

  def histogram(self, column):
    return super().histogram(column)
  
  def pairplot(self):
    pass
