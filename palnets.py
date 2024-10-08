import pandas as pd
import numpy as np
from utils.dataframe import nrows
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from utils.numeric import trimmed_mean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import KNNImputer, MissingIndicator
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from utils.ml.elbow_point import elbow_point

def create_column_transformer(columns):

  categorical = [
    'list', 'discovery_method',
    'last_update', 'name', 'description'
  ]

  numeric = columns.difference(categorical)

  return make_column_transformer(
    (StandardScaler(), numeric),
    (OneHotEncoder(), categorical)
  )

planets = pd.read_csv("data/planets/planets.csv")
planets_u = pd.read_csv("data/planets/planets_updated.csv")


for df in [planets, planets_u]:
  df.rename(
    columns={
      "periastrontime": "periastron_method",
      "semimajoraxis": "semimajor_axis",
      "discoveryyear": "discovery_year",
      "discoverymethod": "discovery_method",
      "lastupdate": "last_update",
    },
    inplace=True
  )

# analyze planets based on their orbits
params = planets[[
  'mass', 'semimajor_axis',
  'eccentricity', 'period',
  'periastron_method', 'periastron'
]]

# investigate the correlation 
# between the planetary parameters
fig = plt.figure(figsize=(10, 8))
plt.suptitle('Orbital parameters of planets')
corr = params.corr(method='pearson') # pearson correlation coefficient

for ax in ['index', 'columns']:
  corr.rename(
    lambda x: x.replace("_", " "),
    axis=ax, inplace=True
  )

heatmap = sns.heatmap(
  corr, 
  annot=True, 
  center=0,
  vmin=-1, # minimum possible correlation coefficient
  vmax=1,  # maximum possible correlation coefficient
  square=True,
  cbar_kws={'shrink': 0.8}
)
plt.show()

# result: 
# -------
# the mass of planets weakly correlates with ecentricity (0.21)
# eccentricity and semimajor axis are also weakly correlated (0.16)
# the semimajor axis and period are strongly correlated (0.97)
# the mass and period are weakly correlated (0.041) and so is
# the eccentricity and period (0.1)

planets.eccentricity.hist(bins=20)
plt.xlabel('Eccentricity')
plt.ylabel('Frequency')
plt.title('Orbit eccentricities')
plt.show()

# separate the parameters with strongest correlation
subset = planets[['semimajor_axis', 'period', 'mass', 'eccentricity']]

# lets investigate the relationshop between the strongest correlated parameters
# semimajor axis and period, these parameters may be erlated to the origin of
# the planets - some come from a Solar System, others from other systems
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
in_solar_system = (planets.list == 'Solar System')
scatter = sns.scatterplot(
  x=planets.semimajor_axis,
  y=planets.period, 
  hue=in_solar_system,
  ax=ax
)

ax.set_xlabel('semimajor axis')
ax.set_ylabel('log period (days)')
ax.set_yscale('log')

solar_system = planets[planets.list == 'Solar System']

for planet in solar_system.name:
  data = solar_system.query(f'name == "{planet}"')
  ax.annotate(
    planet,
    (data.semimajor_axis, data.period),
    (7 + data.semimajor_axis, data.period),
    arrowprops=dict(arrowstyle='->')
  )

ax.set_title('Planetary period vs. semimajor axis')
plt.legend(title='in solar system?', loc='lower right')
plt.show()

# prep data for machine learning
x = planets[['eccentricity', 'semimajor_axis', 'mass']] # independent feats?
y = planets['period'] # dependent feats

x_train, x_test, y_train, y_test = train_test_split(
  x, y,
  test_size=0.25, # ideally between 10-30% of the data
  random_state=0  # for reproducibility of shuffling
)

# NOTE: in big data scenarios, there will most likely be used 
# much less than 70% of it for training because the computational
# costs may rise significantly for possibly minuscule improvements 
# and an increased risk of overfitting

# normalize the data using Z-score normalization;
# alternatively, the RobustScaler class, which uses the
# median and IQR (robust to outliers) scaling can be used
x_train_norm = StandardScaler().fit_transform(x_train)
x_test_norm = StandardScaler().fit_transform(x_test)

# one-hot encode the list column
dummies = pd.get_dummies(planets.list, drop_first=True)

# some models may be significantly affected by the high correlation 
# between one-hot columns due to multicollinearity, so we will remove
# one redundant column -  the values in the remaining ones can be used 
# to determine the value for the removed column
missing_flags = MissingIndicator().fit_transform(
  planets[['semimajor_axis', 'mass', 'eccentricity']]
)

grp_data = planets[['semimajor_axis', 'period']].dropna()
n_clusters = 8 
# NOTE: the number of clusters is experimental only, without 
# any domain kowledege it is hard to determine the optimal
# number of clusters; the Elbow method can be used to determine
# the optimal number of clusters, but it is not always accurate.
# The number of clusters was selected to be equal to the number
# of planets in the Solar System.

pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('kmeans', KMeans(n_clusters, random_state=0)),
])
pipeline.fit(grp_data)
prediction = pipeline.predict(grp_data)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
  x='semimajor_axis', y='period',
  hue=prediction,
  data=grp_data, ax=ax,
  palette='Accent'
)
ax.set_yscale('log')

solar_system = planets[planets.list == 'Solar System']

for planet in solar_system.name:
  data = solar_system.query(f'name == "{planet}"')
  ax.annotate(
  planet,
  (data.semimajor_axis, data.period),
  (7 + data.semimajor_axis, data.period),
  arrowprops=dict(arrowstyle='->')
)
  
ax.get_legend().remove()
ax.set_title('KMeans Clusters')
plt.show()

# from the experimental clustering we can see that the number of
# clusters is not optimal, as the planets in the Solar System are
# not clustered together; we can try the Elbow method to determine
# the optimal number of clusters and check the result in a scatterplot
ax = elbow_point(grp_data, pipeline)
ax.annotate(
  'possible appropriate values for k', xy=(2, 900),
  xytext=(2.5, 1500), arrowprops=dict(arrowstyle='->')
)
ax.annotate(
  '', xy=(3, 3480), xytext=(4.4, 1450),
  arrowprops=dict(arrowstyle='->')
)
plt.show()


# the Elbow method suggests that the optimal number of 
# clusters is 3 so let's try to cluster the planets again
n_clusters = 3
pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('kmeans', KMeans(n_clusters, random_state=0)),
])
pipeline.fit(grp_data)
prediction = pipeline.predict(grp_data)
print("Cluster centers:\n", pipeline['kmeans'].cluster_centers_)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
  x='semimajor_axis', y='period',
  hue=prediction,
  data=grp_data, ax=ax,
  palette='Accent'
)
ax.set_yscale('log')

solar_system = planets[planets.list == 'Solar System']

for planet in solar_system.name:
  data = solar_system.query(f'name == "{planet}"')
  ax.annotate(
  planet,
  (data.semimajor_axis, data.period),
  (7 + data.semimajor_axis, data.period),
  arrowprops=dict(arrowstyle='->')
)
  
ax.get_legend().remove()
ax.set_title('KMeans Clusters')
plt.show()


breakpoint()
pass