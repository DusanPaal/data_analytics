"""Utility functions for stock analysis."""

from functools import wraps
import re
from os import path

import pandas as pd

def _sanitize_label(label: str) -> str:
  """Clean up a label by removing non-letter, non-space
  characters and putting in all lowercase with underscores
  replacing spaces.

  Parameters:
  -----------
  label: str
    The text to be fixed.

  Returns:
  --------
  str
  The sanitized label.
  """

  # remove leading and trailing empty spaces
  label = label.lower().strip()

  # replace any duplicate spaces in  
  # the label text with a single space
  label = re.sub(r'\s+', ' ', label)

  # remove non-letter and non-space characters
  label = re.sub(r'[^\w\s]', '', label)

  # replace spaces with underscores
  label = label.replace(' ', '_')

  return label

def label_sanitizer(method):
  """Decorator around a method that returns a dataframe 
  to clean up all labels in said dataframe (column names 
  and index name) by using `_sanitize_label()`.
  
  Parameters:
  -----------
  method: 
    The method to wrap.

  Returns:
  --------
  A decorated method or function.
  """

  @wraps(method) # keeps the original docstrig for help()
  def method_wrapper(self, *args, **kwargs):
    """Wrap the method to sanitize the labels."""

    # call the original method
    df = method(self, *args, **kwargs)

    # fix the column names
    df.columns = [_sanitize_label(col) for col in df.columns]

    # fix the index name
    df.index.name = _sanitize_label(df.index.name)

    return df
  
  return method_wrapper

def group_stocks(mapping: dict) -> pd.DataFrame:
  """Create a new dataframe with many assets and a new 
  column indicating the asset that row's data belongs to.

  Parameters:
  -----------
  mapping: 
  A key-value mapping of the form {asset_name: asset_df}

  Returns:
  --------
  A new `pandas.DataFrame` object.
  """

  group_df = pd.concat(
    [
        stock_data.copy(deep=True).assign(name=stock)
        for stock, stock_data in mapping.items()
    ],
    sort=True
  )
  group_df.index = pd.to_datetime(group_df.index)

  return group_df


def validate_dataframe(columns, instance_method=True):
  """Decorator that raises a `ValueError` if input isn't
  a `DataFrame` or doesn't contain the proper columns. Note
  the `DataFrame` must be the first positional argument
  passed to this method.

  Parameters:
  -----------
  columns: 
    A set of required column names.
    For example, {'open', 'high', 'low', 'close'}.

  instance_method: 
    Whether or not the item being decorated
    is an instance method. Pass `False to
    decorate static methods and functions.

  Returns:
  --------
  A decorated method or function.
  """

  def method_wrapper(method):
    """Wrap the method to sanitize the labels."""

    @wraps(method)
    def validate_wrapper(self, *args, **kwargs):
      # functions and static methods don't pass self so
      # self is the 1st positional argument in that case
      df = (self, *args)[0 if not instance_method else 1]

      if not isinstance(df, pd.DataFrame):
        raise ValueError('The first argument must be a pandas.DataFrame!')

      if columns.difference(df.columns):
        raise ValueError(
          f'Dataframe must contain the following columns: {columns}'
        )

      return method(self, *args, **kwargs)

    return validate_wrapper
  
  return method_wrapper

@validate_dataframe(columns={'name'}, instance_method=False)
def describe_group(data: pd.DataFrame) -> pd.DataFrame:
  """Run `describe()` on the asset group.

  Parameters:
  -----------
  data:
  Grouped data resulting from `group_stocks()`

  Returns:
  --------
  The transpose of the grouped description statistics.
  """
  return data.groupby('name').describe().T

@validate_dataframe(columns=set(), instance_method=False)
def make_portfolio(data, date_level='date'):
  """Make a portfolio of assets by grouping 
  by date and summing all columns.

  Note: 
  -----
  The caller is responsible for making sure the
  dates line up across assets and handling when they don't.
  """
  return data.groupby(date_level).sum()
