import numpy as np 
import pandas as pd
from scipy import stats

def generate_random_numbers(n, dist, seed=None) -> str:

  if seed:
    np.random.seed(seed)
    
  if dist == "uniform":
    return np.random.uniform(size=n)
  
  if dist == "normal":
    return np.random.normal(size=n)
  
  if dist == "exponential":
    return np.random.exponential(size=n)
  
  if dist == "poisson":
    return np.random.poisson(size=n)
      
  raise ValueError(f"Invalid distribution: {dist}")

def generate_number_series(n, method) -> np.array:
   
  if method == "linspace":
    return np.linspace(0, 10, n)
   
  if method == "range":
    return np.arange(n)
  
def trimmed_mean(vals, ndigits=2, pct_trim=0.1, coerce='float'):

  if isinstance(vals, pd.Series):
    vals = vals.dropna()

  tm = stats.trim_mean(vals, pct_trim)
  tm = round(tm, ndigits)

  if coerce == 'int':
    return int(tm)
  if coerce == 'float':
    return float(tm)
