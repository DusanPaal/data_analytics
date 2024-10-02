"""
Analysis of varied parameters of earthquakes 
and their relationship to the occurence of 
tsunami waves.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plots import show_plot

# load the data from the csv file
data = pd.read_csv('data/earthquakes/quakes.csv')

# rename the columns to more readable names
data.rename(columns={
  "parsed_place": "country",
  "mag": "magnitude",
  "magType": "magnitude_type",
  "place": "place"
}, inplace=True)

# convert the columns to appropriate data types
data.tsunami = data.tsunami.astype(bool)
data.time = pd.to_datetime(data.time, unit='ms', origin = 'unix')

# create strip plot to understand the magnitude types,
# the range of magnitudes, and their relationship to 
# the occurence of tsunami waves
sns.stripplot(
  x='magnitude_type',
  y='magnitude', 
  hue='tsunami',
  data=data.query('country == "Indonesia"')
)
show_plot('Comparison of earthquake magnitude by magnitude type')


# use the swarmplot to reduce overlaping 
# of the points observed in the strip plot
sns.swarmplot(
  x='magnitude_type',
  y='magnitude', 
  hue='tsunami',
  data=data.query('country == "Indonesia"'),
  size=3.5 # point size
)
show_plot('Comparison of earthquake magnitude by magnitude type')


# create an enhanced boxplot to understand 
# the quantile fractions of the earthquake 
# magnitudes for different magnitude types
sns.boxenplot(
  x='magnitude_type',
  y='magnitude', 
  data=data[['magnitude_type', 'magnitude']],
)
show_plot('Comparison of earthquake magnitude by magnitude type')

# create a violin plot to understand the distribution
# of the earthquake magnitudes by the magnitude type
# by comnining a kernel density estimate and a box plot
fig, axes = plt.subplots(figsize=(10, 5))
sns.violinplot(
  x='magnitude_type',
  y='magnitude', 
  data=data[['magnitude_type', 'magnitude']],
  ax=axes, scale='width' #all violins will have the same width
)
show_plot('Comparison of earthquake magnitude by magnitude type')

# alternatively, we can use countplot or barplot to
# visualize the frequency of occurence of each magnitude

# countplot
fig, axes = plt.subplots(figsize=(10, 5))
sns.countplot(
  x='magnitude_type',
  data=data[['magnitude_type']],
  ax=axes
)
show_plot('Comparison of earthquake magnitude by magnitude type')

# barplot to show the average magnitude of each magnitude 
# type and the variance in the magnitude values
sns.barplot(
  x='magnitude_type',
  y='magnitude', 
  data=data[['magnitude_type', 'magnitude']],
)
show_plot('Comparison of earthquake magnitude by magnitude type')

# analyze the earthquakes in Indonesia and Papua New Guinea
facet_grid = sns.FacetGrid(
  data.query(
    'country.isin('
    '["Indonesia", "Papua New Guinea"]'
    ') and magnitude_type == "mb"'),
  row='tsunami',
  col='country',
  height=4
)

facet_grid.map(sns.histplot, 'magnitude', kde=True)
show_plot()

breakpoint()
pass