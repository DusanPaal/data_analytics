"""Utility functions for stock analysis."""

from functools import wraps
import re

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
  label = re.sub(r's+', ' ', label)

  # remove non-letter and non-space characters
  label = re.sub(r'\W+', '', label)

  # replace spaces with underscores
  label.replace(' ', '_')

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
