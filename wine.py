"""
Analysis of wine quality based on data as published in a research paper:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. 
Decision Support Systems, Elsevier, 47(4):547-553, 2009. 
Data source available at http://archive.ics.uci.edu/ml/datasets/Wine+Quality
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kstest, shapiro
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def force_fullwidth_output():
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 1000)

def describe(data, n_decimals=2):
  """Print descriptive statistics for the data."""

  desc = data.describe().apply(
    lambda x: round(x, n_decimals)
  ).T
  desc['count'] = desc['count'].astype(int)
  print(desc)

def plot_quality_score(data, kind):
  """Plot the distribution of wine quality."""

  sorted = data.quality.value_counts().sort_index()
  ax = sorted.plot.barh(
    title=f'{kind.title()} - Wine Quality Scores',
    figsize=(12, 3)
  )
  ax.axes.invert_yaxis()

  for bar in ax.patches:
    # number of wines with this quality score
    # as a percentage of the total number of wines
    offset_pts = 5
    percent = bar.get_width() / data.shape[0]
    pos_vertical_center = bar.get_y() + bar.get_height() / 2
    ax.text(
      bar.get_width() + offset_pts, 
      pos_vertical_center,
      f'{percent:.1%}', 
      va='center'
    )

  plt.xlabel('Count of wines')
  plt.ylabel('Quality score')

  # hide the top and right spines
  for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

  plt.show()

def plot_chemical_props(data):
  """Plot the chemical properties of the wine."""

  # examine each chemical property by visualizing in box plots
  chemical_props = set(data.columns).difference(['quality', 'high_quality', 'kind'])
  melted = data.drop(columns='quality').melt(id_vars=['kind'])

  fig, axes = plt.subplots(
    math.ceil(len(chemical_props) / 4), 4, figsize=(15, 10)
  )

  for ax, prop in zip(axes.flatten(), chemical_props):
    sns.boxplot(
      x='variable',
      y='value', 
      hue='kind',
      data=melted[melted.variable == prop],
      ax=ax
    ).set_xlabel('')

    # ax.set_title(f'{prop.title()} by wine type')
    # ax.set_xlabel('')

  for ax in axes[len(chemical_props):]:
    ax.remove() # remove empty subplots

  plt.suptitle('Chemical properties by wine type')
  plt.tight_layout()
  plt.show()

def plot_quality_scores(red_wine, white_wine):
  """Plot the quality scores of the wine."""

  plot_quality_score(red_wine, 'red wine')
  plot_quality_score(white_wine, 'white wine')


# read wine data
red_wine = pd.read_csv("data/wine/winequality-red.csv", sep=';')
white_wine = pd.read_csv("data/wine/winequality-white.csv", sep=';')

# examine the distribution of quality scores for both kinds of wine 
plot_quality_scores(red_wine, white_wine)

# generate random normally distributed data with sample size 1000
rnd = np.random.normal(loc=0, scale=1, size=1000)

# check for normality
for kind, data in [('red', red_wine), ('white', white_wine)]:
  print(f'{kind.title()} wine quality scores:')
  val, p = shapiro(data.quality)
  print(f'Shapiro-Wilk test: {round(val, 3)} (p-value: {round(p,5)})')
  val, p = kstest(data.quality, 'norm')
  print(f'Kolmogorov-Smirnov test: {round(val, 3)} (p-value: {round(p,5)})')

# categorize wine quality as low or high
red_wine['high_quality'] = pd.cut(
  red_wine.quality,
  bins=[0, 6, 10],
  labels=[0, 1]
)

white_wine['high_quality'] = pd.cut(
  white_wine.quality,
  bins=[0, 6, 10],
  labels=[0, 1]
)

wine = pd.concat([
  white_wine.assign(kind='white'),
  red_wine.assign(kind='red')
])

plot_chemical_props(wine)

x = wine[[
  'chlorides',
  'total sulfur dioxide',
  'free sulfur dioxide',
  'volatile acidity',
  'sulphates',
  'density',
  'fixed acidity'
  # these feats exhibit the most variance
]]

y = wine['high_quality']

# prep data for machine learning
x_train, x_test, y_train, y_test = train_test_split(
  x, y,
  test_size=0.25, # ideally between 10-30% of the data
  random_state=0  # for reproducibility of shuffling
)

# encode the kind of wine as a binary variable
encoded = np.where(wine.kind == 'red', 1, 0)
binned = pd.cut(red_wine.quality, bins=[-1, 3, 6, 10], labels=['0-3 (low)', '4-6 (med)', '7-10 (high)'])
encoded = LabelEncoder().fit_transform(binned)
converted = pd.Series(encoded, name='quality')

breakpoint()
pass
