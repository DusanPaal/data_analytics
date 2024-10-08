def nrows(*dfs: list) -> list:
  """Returns the number of rows for each DataFrame provided.

  Parameters:
  -----------
  *dfs (DataFrame): 
    One or more pandas DataFrame objects.

  Returns:
  --------
  list: A list containing the number of rows for each DataFrame.
  """


  return [df.shape[0] for df in dfs]

