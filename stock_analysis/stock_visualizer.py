"""Visualize financial instruments."""

# Standard library imports
import math
from typing import Callable

# Third-party library imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

# Local library imports
from .utils import validate_dataframe

class Visualizer:
  """Base class for specific visualizers."""

  @validate_dataframe(columns={'open', 'high', 'low', 'close'})
  def __init__(self, df) -> None:
    """Store the input data as an attribute."""

    self.data = df

  @staticmethod
  def _iter_handler(items):
    """Static method for making a list out of 
    an item if it isn't a list or tuple already.

    Parameters:
    -----------
    items: 
      The variable to make sure it is a list.

    Returns:
    --------
    The input as a list or tuple.
    """
    return items if isinstance(items, (list, tuple)) else [items]
  
  def _window_calc(
      self, column, periods, name, 
      unc: Callable, named_arg, **kwargs
    ) -> pd.DataFrame:
    """To be implemented by subclasses. Defines how 
    to add lines resulting from window calculations.
    """
    raise NotImplementedError('Subclasses must implement this method!')
  
  def _shade_region(self, ax, shader):
    """Static method for shading a region on a plot.

    Parameters:
    -----------
    ax: 
      `Axes` object to add the shaded region to.

    x: 
      Tuple with the `xmin` and `xmax` bounds 
      for the rectangle drawn vertically.

    y:
      Tuple with the `ymin` and `ymax` bounds 
      for the rectangle drawn horizontally.

    kwargs: 
      Additional keyword args. to pass down.

    Returns:
    --------
    The matplotlib `Axes` object passed in.
    """

    x = shader.get('x', tuple())
    y = shader.get('y', tuple())
    kwargs = shader.get('kwargs', dict())

    if not (x or y):
      raise ValueError(
        'Either x or an y min/max tuple must be provided!')
    
    if x and y:
      raise ValueError(
        'Both x and y cannot be provided at the same time!')
    
    if x and not y:
      ax.axvspan(*x, **kwargs) # vertical rectangle
    elif not x and y:
      ax.axhspan(*y, **kwargs) # horizontal rectangle
  
  def moving_average(
      self, column, periods,
      title=None, ylabel=None,
      shader=None, **kwargs):
    """Add line(s) for the moving average of a column.

    Parameters:
    -----------
    column:
      The name of the column to plot.

    periods:
      The rule or list of rules for
      resampling, like '20D' for 20-day periods.

    kwargs:
      Additional arguments to pass down.

    Returns:
    --------
    A matplotlib `Axes` object.
    """

    ax = self._window_calc(
      column, periods, 
      name='MA', 
      named_arg='rule', 
      func=pd.Series.resample, 
      **kwargs
    )

    ax.set(title=title, ylabel=ylabel)

    if shader:
      self._shade_region(ax, shader)

    plt.show()

  def exp_smoothing(
      self, column, periods, 
      title=None, ylabel=None, 
      shader=None, **kwargs):
    """Add line(s) for the exponentially 
    smoothed moving average of a column.

    Parameters:
    -----------
    column: 
      The name of the column to plot.

    periods: 
      The span or list of spans for,
      smoothing like 20 for 20-day periods.

    kwargs:
      Additional arguments to pass down.

    Returns:
    --------
    A matplotlib `Axes` object.
    """

    ax = self._window_calc(
      column, periods, 
      name = 'EWMA', 
      func=pd.Series.ewm, 
      named_arg='span',
      **kwargs
    )

    ax.set(title=title, ylabel=ylabel)

    if shader:
      self._shade_region(ax, shader)

    plt.show()
  
  def region_shader(self, x=tuple(), y=tuple(), **kwargs):
    """Compile the region shader for the plot."""
    return{'x': x, 'y': y, 'kwargs': kwargs}

  @staticmethod
  def reset_plottig():
    """Reset the plotting configuration."""
    mpl.rcdefaults()
  
  @staticmethod
  def add_reference_line(ax, x=None, y=None, **kwargs):
    """Static method for adding reference lines to plots.

    Parameters:
    -----------
    ax:
      An `Axes` object to add the reference line to.

    x, y: 
      The x, y value to draw the line at as a
      single value or numpy array-like structure:
        - For horizontal: pass only `y`
        - For vertical: pass only `x`
        - For AB line: pass both `x` and `y`

    kwargs:
      Additional keyword args. to pass down.

    Returns:
    --------
    The matplotlib `Axes` object passed in.
    """

    try:
      # numpy array-like structures -> AB line
      if x.shape and y.shape:
        ax.plot(x, y, **kwargs)
    except Exception as e:
      # error triggers if x or y isn't array-like
      try:
        if x and not y:
          ax.axvline(x, **kwargs) # vertical line
        elif not x and y:
          ax.axhline(y, **kwargs) # horizontal line
        else:
          raise ValueError('Either x or y must be provided!')
      except Exception as exc:
        raise ValueError(
          'If providing only `x` or `y`, '
          'it must be a single value'
        ) from exc
      
    ax.legend()

    return ax
  
  # abstract methods for subclasses to define
  def evolution_over_time(self, column, **kwargs):
    """To be implemented by subclasses for generating line plots."""
    raise NotImplementedError('To be implemented by subclasses.')

  def boxplot(self, **kwargs):
    """To be implemented by subclasses for generating box plots."""
    raise NotImplementedError('To be implemented by subclasses.')

  def histogram(self, column, **kwargs):
    """To be implemented by subclasses for generating histograms."""
    raise NotImplementedError('To be implemented by subclasses.')

  def after_hours_trades(self):
    """To be implemented by subclasses for showing the effect of after-hours trading."""
    raise NotImplementedError('To be implemented by subclasses.')

  def pairplot(self, **kwargs):
    """To be implemented by subclasses for generating pairplots."""
    raise NotImplementedError('To be implemented by subclasses.')
    

class StockVisualizer(Visualizer):
  """Visualizer for a single stock."""

  def _window_calc(
      self, column, periods, name, 
      func, named_arg, **kwargs):
    """Helper method for plotting a series and adding
    reference lines using a window calculation.

    Parameters:
    -----------
    column: 
      The name of the column to plot.

    periods: 
      The rule/span or list of them to pass to
      the resampling/smoothing function, like 
      '20D' for 20-day periods (resampling) or 
      20 for a 20-day span (smoothing)

    name: 
      The name of the window calculation 
      (to show in the legend).

    func: 
      The window calculation function.

    named_arg: 
      The name of the argument `periods`
      is being passed as.

    kwargs: 
      Additional arguments to pass down.

    Returns:
    --------
    A matplotlib `Axes` object.
    """

    ax = self.data.plot(y=column, **kwargs)

    for period in self._iter_handler(periods):
      self.data[column].pipe(func, **{named_arg: period}).mean().plot(
        ax=ax, linestyle='--', label=f"""{period if isinstance(
          period, str) else str(period) + 'D'} {name}"""
      )

    plt.legend()

    return ax

  def evolution_over_time(self, column, **kwargs):
    """Visualize the evolution over time of a column.
    
    Parameters:
    -----------
    column: 
      The name of the column to visualize.

    kwargs:
      Additional arguments to pass down.

    Returns:
    --------
    A matplotlib `Axes` object.
    """
    return self.data.plot.line(y=column, **kwargs)

  def candlestick(self, date_range=None, resample=None, volume=False, **kwargs):
    """Create a candlestick plot for the OHLC data with 
    optional aggregation, subset of the date range, and volume.

    Parameters:
    -----------
    date_range:
      String or `slice()` of dates to pass
      to `loc[]`, if `None` the plot will be
      for the full range of the data.

    resample:
      The offset to use for resampling
      the data, if desired.

    volume:
      Whether to show a bar plot for volume
      traded under the candlesticks.

    kwargs:
      Keyword args for `mplfinance.plot()`.

    Note:
    -----
    The `mplfinance.plot()` doesn't return anything.
    To save your plot, pass in `savefig=file.png` in kwargs.
    """

    if not date_range:
      date_range = slice(
        self.data.index.min(),
        self.data.index.max()
      )

    plot_data = self.data.loc[date_range]

    if resample:
      agg_dict = {
        'open': 'first', 
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
      }

      plot_data = plot_data.resample(resample).agg({
        col: agg_dict[col] for col in plot_data.columns
        if col in agg_dict
      })

    mpf.plot(plot_data, type='candle', volume=volume, **kwargs)
    plt.show()

  def correlation_heatmap(self, other):
    """Plot the correlations between this 
    asset and another one with a heatmap.

    Parameters:
    -----------
    other: 
      The other dataframe.

    Returns:
    --------
    A seaborn heatmap.
    """

    corrs = self.data.pct_change().corrwith(other.pct_change())
    corrs = corrs[~pd.isnull(corrs)]
    size = len(corrs)
    matrix = np.zeros((size, size), float)

    for i, corr in zip(range(size), corrs):
      matrix[i][i] = corr

    # create mask to only show diagonal
    mask = np.ones_like(matrix)
    np.fill_diagonal(mask, 0)

    heatmap = sns.heatmap(
      matrix, 
      annot=True, 
      center=0, 
      vmin=-1,
      vmax=1,
      mask=mask, 
      xticklabels=self.data.columns,
      yticklabels=self.data.columns
    )

    plt.show()
  
  @staticmethod
  def fill_between(
      y1, y2, title, label_higher, 
      label_lower, figsize, legend_x):
    """Visualize the difference between assets.

    Parameters:
    -----------
    y1, y2:
      Data to plot, filling y2 y1.

    title:
      The title for the plot.

    label_higher:
      Label for when y2 > y1.

    label_lower:
      Label for when y2 <= y1.

    figsize:
      (width, height) for the plot dimensions.

    legend_x:
      Where to place legend below the plot.

    Returns:
    --------
    A matplotlib `Axes` object.
    """

    is_higher = y2 - y1 > 0
    fig = plt.figure(figsize=figsize)

    for exclude_mask, color, label in zip(
      (is_higher, np.invert(is_higher)),
      ('g', 'r'),
      (label_higher, label_lower)
    ):
      plt.fill_between(
        y2.index, y2, y1, figure=fig,
        where=exclude_mask, color=color, label=label
      )
        
    plt.suptitle(title)
    plt.legend(
      bbox_to_anchor=(legend_x, -0.1),
      framealpha=0, ncol=2
    )
      
    for spine in ['top', 'right']:
      fig.axes[0].spines[spine].set_visible(False)

    return fig.axes[0]

  def fill_between_other(self, other_df: pd.DataFrame, figsize=(10, 4)):
    """Visualize difference in closing price between assets.

    Parameters:
    -----------
    other_df:
      The other asset's data.

    figsize: 
      (width, height) for the plot.

    Returns:
    --------
    A matplotlib `Axes` object.
    """

    ax = self.fill_between(
      other_df.open, 
      self.data.close, 
      figsize=figsize,
      legend_x=0.7, 
      label_higher='asset is higher',
      label_lower='asset is lower',
      title='Differential between asset price (this - other)'
    )

    ax.set_ylabel('price')

    return ax
  
  def open_to_close(self, figsize):
    """Visualize the daily change 
    in price from open to close.

    Parameters:
    -----------
    figsize: 
      (width, height) of plot
      exploratory data analysis 429

    Returns:
    --------
    A matplotlib `Axes` object.
    """

    ax = self.fill_between(
      self.data.open,
      self.data.close,
      figsize=figsize,
      legend_x=0.67,
      title='Daily price change (open to close)',
      label_higher='price rose',
      label_lower='price fell'
    )

    ax.set_ylabel('price')

    return ax
  
  def jointplot(self, other, column, **kwargs):
    """Generate a seaborn jointplot for given 
    column in his asset compared to another asset.

    Parameters:
    -----------
    other:
      The other asset's dataframe.

    column:
      Column to use for the comparison.ň

    kwargs:
      Keyword arguments to pass down.

    Returns:
    --------
    A seaborn jointplot.
    """

    sns.jointplot(
      x=self.data[column], y=other[column], **kwargs
    )

    plt.show()

  def pairplot(self, **kwargs):
    """Generate a seaborn pairplot for this asset.

    Parameters:
    -----------
    kwargs:
      Keyword arguments to pass down to `sns.pairplot()`

    Returns:
    --------
    A seaborn pairplot
    """
    return sns.pairplot(self.data, **kwargs)

  def histogram(self, column, **kwargs):
    """Generate the histogram of a given column.

    Parameters:
    -----------
    column:
      The name of the column to visualize.

    kwargs:
      Additional keyword arguments to 
      pass down to the plotting function.

    Returns:
    --------
    A matplotlib `Axes` object.
    """
    return self.data.plot.hist(y=column, **kwargs)

  def after_hours_trades(self):
    """Visualize the effect of 
    after-hours trading on this asset.

    Returns:
    --------
    A matplotlib `Axes` object.
    """

    after_hours = self.data.open - self.data.close.shift()
    monthly_effect = after_hours.resample('1M').sum()
    _, axes = plt.subplots(1, 2, figsize=(15, 3))

    after_hours.plot(
        ax=axes[0],
        title='After-hours trading\n(Open Price - Prior Day\'s Close)'
    ).set_ylabel('price')

    monthly_effect.index = monthly_effect.index.strftime('%Y-%b')

    monthly_effect.plot(
        ax=axes[1],
        kind='bar',
        title='After-hours trading monthly effect',
        color=np.where(monthly_effect >= 0, 'g', 'r'),
        rot=90
    ).axhline(0, color='black', linewidth=1)

    axes[1].set_ylabel('price')
    
    plt.show()

  def boxplot(self, **kwargs):
    """Generate box plots for all columns.

    Parameters:
    -----------
    kwargs:
      Additional keyword arguments to 
      pass down to the plotting function.

    Returns:
    --------
    A matplotlib `Axes` object.
    """
    return self.data.plot(kind='box', **kwargs)

class AssetGroupVisualizer(Visualizer):
  """visualize multiple assets"""

  def __init__(self, df, group_by='name'):
    "This object keeps track of the group by column."
    super().__init__(df)
    self.group_by = group_by

  def _window_calc(
      self,
      column,
      periods,
      name,
      func,
      named_arg,
      **kwargs
    ):
    """Helper method for plotting a series and adding
    reference lines using a window calculation.

    Parameters:
    -----------
    column:
      The name of the column to plot.

    periods:
      The rule/span or list of them to pass to the 
      resampling/smoothing function, like '20D' for
      20-day periods (resampling) or 20 for a 20-day
      span (smoothing).

    name:
      The name of the window calculation 
      (to show in the legend).

    func:
      The window calculation function.

    named_arg:
      The name of the argument `periods`
      is being passed as.

    kwargs:
      Additional arguments to pass down.

    Returns:
    --------
    A matplotlib `Axes` object.
    """

    fig, axes = self._get_layout()
    zipped = zip(axes, self.data[self.group_by].unique())

    for ax, asset_name in zipped:
      subset = self.data.query(
        f'{self.group_by} == "{asset_name}"'
      )

    ax = subset.plot(
      y=column, ax=ax, label=asset_name, **kwargs
    )

    for period in self._iter_handler(periods):
      subset[column].pipe(func, **{named_arg: period}).mean().plot(
      ax=ax, linestyle='--',
      label=f"""{period if isinstance(
        period, str
      ) else str(period) + 'D'} {name}"""
      )

    ax.legend()
    plt.tight_layout()

    return ax

  def _get_layout(self):
    """Helper method for getting an autolayout of subplots.
    Returns: `Figure` and `Axes` objects to plot with.
    """

    subplots_needed = self.data[self.group_by].nunique()
    rows = math.ceil(subplots_needed / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))

    if rows > 1:
      axes = axes.flatten()

    if subplots_needed < len(axes):
      # remove excess axes from autolayout
      for i in range(subplots_needed, len(axes)):
        # can't use comprehension here
        fig.delaxes(axes[i])

    return fig, axes
  
  def evolution_over_time(self, column, **kwargs):
    """Visualize the evolution over time for all assets.

    Parameters:
    -----------
    column:
    The name of the column to visualize.

    kwargs:
    Additional arguments to pass down.

    Returns:
    --------
    A matplotlib `Axes` object.
    """

    if 'ax' not in kwargs:
      _, ax = plt.subplots(
        nrows=1, ncols=1, 
        figsize=(10, 4)
      )
    else:
      ax = kwargs.pop('ax')

    sns.lineplot(
      x=self.data.index, y=column, hue=self.group_by,
      data=self.data, ax=ax, **kwargs
    )

    
  
  def after_hours_trades(self):
    """Visualize the effect of after-hours trading.

    Returns:
    --------
    A matplotlib `Axes` object.
    """
    num_categories = self.data[self.group_by].nunique()

    _, axes = plt.subplots(
      num_categories, 2, figsize=(15, 3 * num_categories)
    )

    zipped = zip(axes, self.data.groupby(self.group_by))

    for ax, (name, data) in zipped:

      after_hours = data.open - data.close.shift()
      monthly_effect = after_hours.resample('1M').sum()

      after_hours.plot(
        ax=ax[0],
        title=f'{name} Open Price - Prior Day\'s Close'
      ).set_ylabel('price')
    
      monthly_effect.index = monthly_effect.index.strftime('%Y-%b')

      monthly_effect.plot(
        ax=ax[1], kind='bar', rot=90,
        color=np.where(monthly_effect >= 0, 'g', 'r'),
        title=f'{name} after-hours trading monthly effect'
      ).axhline(0, color='black', linewidth=1)

      ax[1].set_ylabel('price')
  
    plt.tight_layout()
    return axes

  def boxplot(self):
    pass

  def heatmap(self, pct_change=True, **kwargs):
    """Generate a heatmap for correlations between assets.

    Parameters:
    -----------
    pct_change:
      Whether to show the correlations
      of the daily percent change in price.

    kwargs:
    Keyword arguments to pass down.

    Returns:
    --------
    A seaborn heatmap
    """

    pivot = self.data.pivot_table(
      values='close',
      index=self.data.index,
      columns=self.group_by
    )

    if pct_change:
      pivot = pivot.pct_change()

    sns.heatmap(
      pivot.corr(), 
      annot=True, 
      center=0,
      vmin=-1, 
      vmax=1, 
      **kwargs
    )

    plt.show()

  def histogram(self, column):
    return super().histogram(column)
  
  def pairplot(self, **kwargs):
    """Generate a seaborn pairplot
    for this asset group.

    Parameters:
    -----------
    kwargs:
    Keyword arguments to pass down.

    Returns:
    --------
    A seaborn pairplot
    """

    pivotted = self.data.pivot_table(
      values='close',
      index=self.data.index,
      columns=self.group_by
    )
    
    return sns.pairplot(
      pivotted, 
      diag_kind='kde', 
      **kwargs
    )

def get_cycler(cumm_returns):

  bw_viz_cycler = (
    cycler(
      color=[plt.get_cmap('tab10')(x/10)
      for x in range(10)]
    ) + cycler(
      linestyle=['dashed', 'solid', 'dashdot',
      'dotted', 'solid'] * 2
    )
  )
  
  fig, axes = plt.subplots(1, 2, figsize=(15, 5))
  axes[0].set_prop_cycle(bw_viz_cycler)

  for name, data in cumm_returns.items():
    data.plot(
    ax=axes[1] if name == 'Bitcoin' else axes[0],
    label=name, legend=True
    )

  fig.suptitle('Cumulative Returns')
  plt.show()
