import pandas as pd
from utils.plots import show_plot, get_percent_formatter
from matplotlib.ticker import MultipleLocator, PercentFormatter

# url_1 = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
# url_2 = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
# data = pd.read_csv(url_1)

# read data from the file
data = pd.read_csv('data/covid19/cases.csv')

# prep data for analysis
data = data.assign(
  date = pd.to_datetime(data.dateRep, format='%d/%m/%Y')
)

data.countriesAndTerritories = data.apply(
  lambda x: x.countriesAndTerritories.replace(
    'United_States_of_America', 'USA'
  ), axis=1
)

new_cases = data.pivot(
  index='date',
  columns='countriesAndTerritories',
  values='cases'
)

new_cases.sort_index(axis=1, inplace=True) # sort the records by date
new_cases.fillna(0, inplace=True) # missing values are considered as 0 cases

percent_new_cases = new_cases.apply(
  lambda x: x / new_cases.apply('sum', axis=1), axis=0
)

subset = percent_new_cases.loc[
  :, ['Italy', 'China', 'Spain', 'USA', 'India', 'Brazil']
]

ax = subset.plot(
  figsize=(12, 7),
  title='Percentage of the World\'s New COVID-19 Cases\n(source: ECDC)',
  style=['-'] * 3 + ['--', ':', '-.']
)

tick_locs = subset.index[subset.index.day == 18].unique()
tick_labels = [loc.strftime('%b %d\n%Y') for loc in tick_locs]

ax.legend(title='Country', framealpha=0.5, ncol=2)
ax.set_xlabel('')
ax.set_ylabel('Percentage of the world\'s COVID-19 cases')
ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

for spine in ['top', 'right']:
  ax.spines[spine].set_visible(False)

show_plot(ticks=tick_locs, labels=tick_labels)

# plot daily new cases in New Zealand
ax = new_cases.New_Zealand['2020-04-18':'2020-09-18'].plot(
  title='Daily new COVID-19 cases in New Zealand\n(source: ECDC)'
)
ax.set(xlabel='', ylabel='new COVID-19 cases')

# the original ticks increment by 2.5, so fix this 
# by setting the major locator to 3 which makes more sense
ax.yaxis.set_major_locator(MultipleLocator(base=3)) 

for spine in ['top', 'right']:
  ax.spines[spine].set_visible(False)

show_plot()

breakpoint()
pass