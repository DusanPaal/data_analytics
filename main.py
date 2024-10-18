import numpy as np
from utils.dataframe import get_row_count
from utils.numeric import generate_random_numbers as gen_rand
import pandas as pd
import datetime as dt
import sqlite3
from utils.api import fetch_data
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns

def test_df_generation():

  data = np.genfromtxt(
    r"data/example_data.csv", 
    delimiter=";", 
    names=True, 
    dtype=None, 
    encoding="utf-8"
  )

  array_dict = {
    col: np.array([row[i] for row in data]) 
    for i, col in enumerate(data.dtype.names)
  }

  # find the maximum value in the "value" column
  max_magnitude_a = max([row[3] for row in data]) # slower method to find the maximum magnitude
  max_magnitude_b = array_dict["mag"].max()       # faster method to find the maximum magnitude
  assert max_magnitude_a == max_magnitude_b

  magnitude = pd.Series(array_dict["mag"], name="Magnitude")
  place = pd.Series(array_dict["place"], name="Place")

  numbers = np.linspace(0, 10, 5)
  x = pd.Series(numbers, name="X")
  y = pd.Series(numbers, name="Y", index=pd.Index([1, 2, 3, 4, 5]))
  # can't align the first element of x and the last element of y (they are both NaN)
  # print(x + y) 

  # create a DataFrame from the array_dict
  # and set the index to be the "time" column
  df = pd.DataFrame(array_dict).set_index("time")
  ser = pd.Series(gen_rand(10, dist="normal", seed=0), name="Random")
  converted = ser.to_frame()

  # create a test dataframe
  n_rows = 5

  df = pd.DataFrame({
      'random': gen_rand(n_rows, dist="normal", seed=0),
      'text': ['hot', 'warm', 'cool', 'cold', None],
      'truth': [np.random.choice([True, False]) for _ in range(n_rows)]
    },
    index=pd.date_range(
      start=dt.date(2019, 4, 21),
      periods=n_rows,
      freq="1D",
      name='date'
    )

  )

  # create dataframe form a list of dictionaries
  # as they are normally privided by an API
  df = pd.DataFrame([
    {'mag': 5.2, 'place': 'California'},
    {'mag': 1.2, 'place': 'Alaska'},
    {'mag': 0.2, 'place': 'California'},
  ],
  index=pd.date_range(
      start=dt.date(2019, 4, 21),
      periods=3,
      freq="1D",
      name='date'
    )
  )

  # create dataframe form a list of tuples
  list_of_tuples = [(n, n**2, n**3) for n in range(5)]
  df_a = pd.DataFrame(list_of_tuples, columns=['n', 'n_squared', 'n_cubed'])

  # create a DataFrame from an array
  df_b = pd.DataFrame(np.array([
    [0, 0, 0],
    [1, 1, 1],
    [2, 4, 8],
    [3, 9, 27],
    [4, 16, 64],
  ]), columns=['n', 'n_squared', 'n_cubed'])


  assert df_a.equals(df_b)

def test_data_reading_from_file():

  # read contents of a CSV file and create a DataFrame
  df = pd.read_csv("data/earthquakes.csv")

  print(df.info())
  print(df.describe())

def test_data_slicing():

  df = pd.read_csv("data/earthquakes.csv")

  columns = ['title', 'time'] + [col for col in df.columns if col.startswith('mag')]
  selection_a = df[columns][100:105]
  selection_b = df[100:105][columns]
  assert selection_a.equals(selection_b)

  df.loc[100:103, "title"] = selection_a.loc[100:103, "title"].str.lower()
  df.iloc[10:15, [19, 8]]
  # indexing by integers goves the same values as indexing by labels
  df.iloc[10:15, 6:10].equals(df.loc[10:14, 'gap':'magType'])

  # select a scalar value by label and by integer
  val_a = df.at[15, 'mag']
  val_b = df.at[15, 8]
  assert val_a == val_b

  outlier_mask = df.mag >= 7.0

  outlier_mag = df.loc[
    outlier_mask, 
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
  ]

  selection_c = df.loc[
    (df.tsunami == 1) & (df.alert == 'red'),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
  ]

  assert selection_c.shape[0] == 1

  critical = df.loc[
    (df.tsunami == 1) | (df.alert == 'red'),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
  ]

  assert critical.shape[0] == 61
  critical.mag.min() # critical tsunamis have higher magnitudes
  critical.mag.max() # than the rest of alerts

  selection_e = df.loc[
    (df.place.str.contains('Alaska')) & (df.alert.notnull()),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
  ]

  assert selection_e.shape[0] == 9

  california = df.loc[
    (df.place.str.contains(r'California|CA')) & (df.mag > 3.8),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
  ]

  assert california.shape[0] == 2

  higlhy_critical = df.loc[(
    df.mag.between(6.0, 7.5),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
  )] # contains magnitudes between 6.0 and 7.5 (inclusive)

  mag_types = df.loc[
    (df.magType.isin(['mw', 'mwb'])),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
  ]

  # find rows with lowest and highest magnitudes
  extremes = df.loc[
    (df.mag.idxmin(), df.mag.idxmax()),
    ['alert', 'mag', 'magType', 'title', 'tsunami', 'type']
  ]

  assert extremes.loc[extremes.mag.idxmin(), "title"].endswith("Alaska")
  assert extremes.loc[extremes.mag.idxmax(), "title"].endswith("Indonesia")


def test_data_modification():

  # read teh data from file
  df = pd.read_csv(
    "data/earthquakes.csv",
    usecols=[
      'time', 'title', 'mag',
      'magType', 'place', 'tsunami'
    ]
  )

  df_copy = df.copy()

  # create a new column with a constant value
  # by broadcasting the value to all rows
  df_copy['source'] = 'USGS API'

  # create another new column with boolean values
  # based on the value of the 'mag' column
  df_copy['mag_negative'] = df_copy.mag < 0

  # there are some place names that contain
  # the whole names of states or the state code only
  countries = df.place.str.extract(r', (.*)$')[0].sort_values().unique()
  countries = df.place.str.extract(r', (.*)$')[0].sort_values().unique()
  assert countries.size == 93

  # remove any other extraneous information,
  # replace the country codes with the full names
  # and place the place names into a separate column
  df_copy['parsed_place'] = df_copy.place.str.replace(
    r'.* of ', '', regex=True         # remove <x> of <x>
  ).str.replace(
    'the ', ''                        # remove "the "
  ).str.replace(
    r'CA$', 'California', regex=True  # fix California
  ).str.replace(
    r'NV$', 'Nevada', regex=True      # fix Nevada
  ).str.replace(
    r'MX$', 'Mexico', regex=True      # fix Mexico
  ).str.replace(
    r' region$', '', regex=True       # fix " region" endings
  ).str.replace(
    'northern ', ''                   # remove "northern "
  ).str.replace(
    'Fiji Islands', 'Fiji'            # line up the Fiji places
  ).str.replace(                      # remove anything else extraneous from start
    r'^.*, ', '', regex=True
  ).str.strip()                       # remove any extra spaces

  enriched = df_copy.assign(
    in_california = df_copy.parsed_place.str.endswith('California'),
    in_alaska = df_copy.parsed_place.str.endswith('Alaska'),
    neither = lambda x: ~(x.in_california | x.in_alaska)
  )

  tsunami = df[df.tsunami == 1]
  no_tsunami = df[df.tsunami == 0]
  concatenated = pd.concat([tsunami, no_tsunami])

  assert concatenated.shape[0] == tsunami.shape[0] + no_tsunami.shape[0]

  # read additional columns from the original file
  # and merge them with the concatenated DataFrame
  # along the columns axis
  additional_cols = pd.read_csv(
    "data/earthquakes.csv",
    usecols=['tz', 'felt', 'ids']
  )

  added = pd.concat([concatenated, additional_cols], axis=1)
  assert added.shape[1] == concatenated.shape[1] + additional_cols.shape[1]

  # artificially create a new clmn to the no_tsunami dataframe
  # and try to merge it with the tsunami DataFrame along the rows axis
  no_tsunami = no_tsunami.assign(
    severity = no_tsunami['mag'].apply(lambda x: 'critical' if x > 6.0 else 'normal')
  )

  # concatenate the two DataFrames along the rows axis
  # the resulting dataframe will have NaN values in the severity
  # column for the tsunami rows originating from the 'tsunami' DataFrame
  concat_no_join = pd.concat([tsunami, no_tsunami], ignore_index=True)
  assert "severity" in concat_no_join.columns

  concat_inner_join = pd.concat([tsunami, no_tsunami], join='inner', ignore_index=True)
  assert "severity" not in concat_inner_join.columns

  try:
    del df_copy['source']
  except KeyError:
    print("Could not delete the 'source' column")

  mag_negative = df_copy.pop('mag_negative')
  assert not mag_negative.empty

  # remove some rows from the dataframe
  df_copy.drop([0, 1, 2], inplace=True)

  #remove some columns from the dataframe
  df_copy.drop(['magType', 'tsunami'], inplace=True, axis=1)
  assert 'magType' not in df_copy.columns
  assert 'tsunami' not in df_copy.columns

def test_data_wrangling():

  wide_df = pd.read_csv(
    "data/wide_data.csv",
    parse_dates=['date']
  )

  long_df = pd.read_csv(
    "data/long_data.csv",
    parse_dates=['date'],
    usecols=['date', 'datatype', 'value']
  )[['date', 'datatype', 'value']] # sort columns

  # for plotting wide forma data we use the matplotlib library
  wide_df.plot(
    x='date', y=['TMAX', 'TMIN', 'TOBS'], kind='line',
    figsize=(15, 5), title='Temperature in NYC in October 2018'
  ).set_ylabel('Temperature in Celsius')
  plt.show()

  # for plotting long format data we use the seaborn library
  sns.set(rc={'figure.figsize': (15, 5)}, style='white')
  ax = sns.lineplot(
    data=long_df, x='date', y='value', hue='datatype'
  )

  ax.set_ylabel('Temperature in Celsius')
  ax.set_title('Temperature in NYC in October 2018')
  plt.show()

  # facet the saborn plot by the datatype
  sns.set(rc={'figure.figsize': (15, 5)}, style='white', font_scale=2)
  grid = sns.FacetGrid(long_df, col='datatype', height=10)
  grid = grid.map(plt.plot, 'date', 'value')
  grid.set_titles(size=25)
  grid.set_xticklabels(rotation=45)
  plt.show()

def test_cleaning_data():

  df = pd.read_csv(
    "data/nyc_temperatures.csv",
    parse_dates=['date']
  )

  df.rename(
    columns={
      'value': 'temp_C', 
      'attributes': 'flags'}, 
    inplace=True
  )
  
  df.rename(str.upper, axis='columns')

  # convert the columns to appropriate data types
  df.loc[:, 'date'] = pd.to_datetime(df.date)

  # load the data again, parse the dates
  # while reading the file, set the index 
  # to be the 'date' column, then localize
  # the timezone to eastern time
  eastern = pd.read_csv(
    "data/nyc_temperatures.csv",
    index_col='date', parse_dates=True
  ).tz_localize('EST')

  # remove the timezone information to avoid a warning 
  # from pandas that the PeriodArray class doesn't have
  # time zone information information loss
  eastern.tz_localize(None)
  
  # convert the index to a PeriodIndex 
  # (stored as a PeriodArray)
  eastern.to_period('Y').index

  # read again the data from the file and rename the columns
  df = pd.read_csv("data/nyc_temperatures.csv").rename(
    columns={'value': 'temp_C', 'attributes': 'flags'}
  )

  new_df = df.assign(
    date=pd.to_datetime(df.date),
    temp_C_whole=lambda x: x.temp_C.astype(int),
    temp_F=lambda x: x.temp_C * 9/5 + 32,
    temp_F_whole=lambda x: x.temp_F.astype(int),
  )

  # convert 'station' and 'datatype' columns to a category type
  categorical_df = new_df.assign(
    station = new_df.station.astype('category'),
    datatype = new_df.datatype.astype('category')
  )

  categorical_df.datatype = pd.Categorical(
    categorical_df.datatype,
    categories=['TMIN', 'TAVG', 'TMAX'], # manually set the order of the categories
    ordered=True
  )

  sorted_df = categorical_df.sort_values(
    by='datatype', ascending=False)
  
  max_temps = sorted_df[sorted_df.datatype == 'TMAX'].sort_values(
    by=['temp_C', 'date'], ascending=[False, True]
  )

  # sort the index of the DataFrame
  max_temps.sort_index(inplace=True)

  # EDA insight into data 
  sorted_df[sorted_df.datatype == 'TMAX'].nlargest(
    n=10, columns='temp_C')

  sorted_df[sorted_df.datatype == 'TMAX'].nsmallest(
    n=10, columns='temp_C')

  # select a sample of rows and sort them by row index, then by column index
  sorted_df.sample(n=5, random_state=0).sort_index(axis=1)
  sorted_df.sample(n=5, random_state=0).sort_index(axis=1)

  # select a range of columns with similar names
  selected_b = sorted_df.loc[:, 'station':'temp_F_whole']

  sorted_df.set_index('date', inplace=True)
  sorted_df.loc['2018']
  sorted_df.loc['2018-Q4']
  # sorted_df.loc['2018-10-09':'2018-10-16'] # toto nefunguje
  sorted_df.loc['2018-Q4'].reset_index() # restore the date column

  # reindexing the DataFrame
  data = pd.read_csv(
    "data/sp500.csv", 
    index_col='date', 
    parse_dates=True
  ).drop(
    columns=['adj_close']
  )

  # add a colummn with the week day name and the month name
  sp500 = data.assign(
    day_of_week=lambda x: x.index.day_name(),
    month_name=lambda x: x.index.month_name()
  )

  bitcoin = pd.read_csv(
    "data/bitcoin.csv", 
    index_col='date', 
    parse_dates=True
  ).drop(
    columns=['market_cap']
  )

  portfolio = pd.concat(
    [sp500, bitcoin], sort=False, join='inner'
  ).groupby(level='date').sum()

  portfolio = portfolio.assign(
    day_of_week=lambda x: x.index.day_name()
  )

  def plot_data(portfolio, title, label):

    # plot the closing price from Q4 2017 through Q2 2018
    ax = portfolio.loc['2017-Q4':'2018-Q2'].plot(
      title = title, y='close', figsize=(15, 5), legend = False,
      linewidth=2, label=label
    )

    # formatting
    ax.set_ylabel('price')
    ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

    for spine in ['top', 'right']:
      ax.spines[spine].set_visible(False)

    # display the plot
    plt.show()

  plot_data(portfolio, 'Bitcoin + S&P 500 value without accounting for different indices', None)

  # correct the missing data on the weekends
  sp500_reindexed = sp500.reindex(bitcoin.index).assign(
    day_of_week=lambda x: x.index.day_name(),
    volume=lambda x: x.volume.fillna(0),
    close=lambda x: x.close.ffill(),
    open=lambda x: np.where(x.open.isnull(), x.close, x.open),
    high = lambda x: np.where(x.high.isnull(), x.close, x.high),
    low = lambda x: np.where(x.low.isnull(), x.close, x.low),
  )

  # recreate the portfolio with the corrected data
  fixed_portfolio = sp500_reindexed + bitcoin

  plot_data(
    fixed_portfolio, 
    title = 'Bitcoin + S&P 500 value without accounting for different indices', 
    label = 'reindexed portfolio of S&P 500 + Bitcoin'
  )

def test_data_reshaping():
    
  long_df = pd.read_csv(
    'data/long_data.csv',
    usecols=['date', 'datatype', 'value']
  ).rename(columns={'value': 'temp_C'}).assign(
    date=lambda x: pd.to_datetime(x.date),
    temp_F=lambda x: (x.temp_C * 9/5) + 32
  )

  long_df.set_index('date').T

  pivoted = long_df.pivot(
    index='date', columns='datatype', values=['temp_C', 'temp_F']
  )

  # since the pivoted table has multilevel
  # column index, we need first select the
  # top level to access the 'datatype' columns
  tmin_F = pivoted['temp_F']['TMIN']

  # both temps are equally informative
  # since the conversion is linear
  pivoted.describe()

  multi_index_df = long_df.set_index((['date', 'datatype']))
  unstacked = multi_index_df.unstack() #move the outermost index level to the columns

  # the unstacked DataFrame is identical 
  # with the pivoted DataFrame in this case
  assert unstacked.equals(pivoted)

  new_row = pd.DataFrame({
    'date': pd.to_datetime(['2018-10-01']),
    'datatype': ['TAVG'],
    'temp_C': [10],
    'temp_F': [50]
  })

  extra_data = pd.concat(
    [long_df, new_row]
  ).set_index(
    ['date', 'datatype']
  ).sort_index()

  # unstacking moves the datatype to the columns, 
  # however, since not all records have the TAVG
  # datatype, the resulting DataFrame will have 
  # NaN values in that column.
  extra_data_unstacked = extra_data.unstack()
  assert extra_data_unstacked["temp_C"]["TAVG"].isna().any()

  # do some dataframe melting
  # tp go from wide to long format
  wide_df = pd.read_csv(
    'data/wide_data.csv',
    parse_dates=['date']
  )

  melted = wide_df.melt(
    id_vars=['date'],
    value_vars=['TMAX', 'TMIN', 'TOBS'],
    value_name='temp_C',
    var_name='datatype'
  )

  # similar to meltin the same reuslt can
  # be achieved by using the stack method
  stacked = wide_df.set_index('date').stack()
  stacked.columns = ['date', 'datatype', 'temp_C']
  assert stacked.shape[0] == melted.shape[0]

  # since the stacked value type is Series
  # it needs to be converted to a DataFrame
  stacked = stacked.to_frame('values')

  # set the index names to 'date' and 'datatype'
  stacked.index.set_names(['date', 'datatype'], inplace=True)

def test_handle_inconsistent_data():

  df = pd.read_csv('data/dirty_data.csv')
  
  null_vals = df[
    df.SNOW.isna() |
    df.SNWD.isna() |
    df.TOBS.isna() |
    df.WESF.isna() |
    df.inclement_weather.isna()
  ]

  def get_inf_count(df):
    return {
      col: df[df[col].isin([-np.inf, np.inf])].shape[0]
      for col in df.columns
    }
  
  # count the infinite values in the DataFrame
  desc = pd.DataFrame({
    'np.inf SNWD': df[df.SNWD == np.inf].SNOW.describe(),
    '-np.inf SNWD': df[df.SNWD == -np.inf].SNOW.describe(),
  })

  df[df.duplicated(subset=['date', 'station'], keep=False)].shape[0]

  # convert the date column to a datetime object
  df.date = pd.to_datetime(df.date)

  # backup the WESF column as a Series
  unknown_wesf_station = df[df.station == '?'].drop_duplicates('date').set_index('date').WESF

  # Sort the dataframe by the station column in descending
  # order to put the station with no ID (?) last:
  df.sort_values('station', ascending=False, inplace=True)

  #remove rows that are duplicated based on date, keep the first
  # occurencewhich will be ones where thes station has an ID
  deduped =  df.drop_duplicates('date')

  # drop the station column and set the index to the date 
  # column so tht it matches the WESF data
  deduped = deduped.drop(columns='station').set_index('date').sort_index()

  # update the WESF column unisng the combine_first() method
  # to coalesce the values to the first non-null entry 
  deduped = deduped.assign(
    WESF = lambda x: x.WESF.combine_first(unknown_wesf_station)
  )
  assert deduped.dropna(how='all').shape[0] == 324

  # if needed, we can drop the rows where specific columns are null
  assert deduped.dropna(
    how='all', 
    subset=['inclement_weather', 'SNOW', 'SNWD']
  ).shape[0] == 293

  # alternatively, we can provide a threshold for the number of null  
  # values that must be observed to drop the data with the thresh argument.
  # if the portion of null vals exceeds 75 % of the rows, the column is dropped
  percent = 0.75 
  deduped.dropna(axis='columns', thresh = deduped.shape[0] * percent)

  # The WESF column contains mostly null values, but since it is a measurement
  # in milliliters that takes on the value of NaN when there is no water 
  # equivalent of snowfall, we can fill in the nulls with zeros.
  deduped.loc[:, 'WESF'].fillna(0, inplace=True)

  # replace the outler temperatires with NA, since they
  # represent missing data
  deduped = deduped.assign(
    TMAX=lambda x: x.TMAX.replace(5505, np.nan),
    TMIN=lambda x: x.TMIN.replace(-40, np.nan)
  )

  # for demonstration purposes, we will replace 
  # the missing values with the next non-null ones
  deduped = deduped.assign(
    TMAX=lambda x: x.TMAX.ffill(),
    TMIN=lambda x: x.TMIN.ffill(),
  )

  # If we want to handle the nulls and infinite values in the SNWD column, we can 
  # turn NaN into 0 and inf/-inf into very large positive/negative finite numbers,
  # making it possible for machine learning models to learn from the data.
  deduped = deduped.assign(
    SNWD=lambda x: np.nan_to_num(x.SNWD)
  )

  # alternatively, clipping can be used to set the SNOW
  # values to zero where they are infinitely small, and 
  # to a an estimated value (from 'SNWD'), where they are 
  # infinitely large
  deduped = deduped.assign(
    SNWD=lambda x: np.clip(x.SNWD, 0, x.SNOW)
  )

  # fill the missing values in the TMAX and TMIN columns
  # with the median values of the respective columns, 
  # then impute missing values in the TOBS column with 
  # the average of the TMIN and TMAX columns
  imputed_a = deduped.assign(
    TMAX=lambda x: x.TMAX.fillna(x.TMAX.median()),
    TMIN=lambda x: x.TMIN.fillna(x.TMIN.median()),
    TOBS=lambda x: x.TOBS.fillna((x.TMIN + x.TMAX) / 2)
  )

  # imputing of the data above gave us inaccurate results
  # since values from the entire month were ised in the 
  # calculations. To fix this, we will use the rolling median
  # of the last 7 days to impute the missing values in the TOBS column
  imputed_b = deduped.apply(
    lambda x: x.fillna(x.rolling(7, min_periods=0).median())
  )

  # even more accurate strategy would be using linear interpolation
  # of the missing dates so that the values are estimated based on
  # the values before and after the missing values
  imputed_c = deduped.reindex(
    pd.date_range('2018-01-01', '2018-12-31', freq='D')
  ).apply(lambda x: x.interpolate()).head(10)


def test_data_reading_from_db():

  with sqlite3.connect("data/quakes.db") as conn:
    tsunamis = pd.read_sql("SELECT * FROM tsunamis", conn)

  assert not tsunamis.empty

def test_data_reading_from_api():
  # fetch data from the USGS API for the last 30 days
  # and store it in a DataFrame

  api = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
  yesterday = dt.date.today() - dt.timedelta(days=1)
  start_time = yesterday - dt.timedelta(days=30)

  payload = {
    'format': 'geojson',
    'starttime': start_time,
    'endtime': yesterday,
  }

  data = fetch_data(api, payload, False)

  print("----------------------")
  print("Data info:")
  print("----------------------")
  for (key, val) in data['metadata'].items():
    if key == 'generated':
      # convert the UNIX timestamp to a datetime object
      val = dt.datetime.utcfromtimestamp(val / 1000)
    print(key, ":", val)
  print("----------------------")

  features = data['features']
  # extract the properties from each feature
  data = [rec['properties'] for rec in features]
  df = pd.DataFrame(data)
  assert not df.empty

def test_data_aggregation(): 

  # read the data from the CSV file
  weather = pd.read_csv(
    'data/nyc_weather_2018.csv', 
    parse_dates=['date']
  ).sort_values(
    ['date', 'station']
  )

  # there shpuld be no duplicated values in 
  # the 'station' and 'datatype' columns
  assert weather[weather.duplicated(keep=False)].empty

  snow_data_q = weather.query(
    'datatype == "SNOW" and value > 0'
    'and station.str.contains("US1NY")'
  )

  snow_data_m = weather[
    (weather.datatype == 'SNOW') & 
    (weather.value > 0) & 
    (weather.station.str.contains('US1NY'))
  ]

  # queries and boolean indexing should return the same results
  assert snow_data_q.equals(snow_data_m)

  stations = pd.read_csv('data/weather_stations.csv')
  stations.rename(columns={'id': 'station'}, inplace=True)

  inner_join = weather.merge(stations, on='station', how='inner')
  right_join = weather.merge(stations, on='station', how='right')
  left_join = stations.merge(weather, on='station', how='left')
  outer_join = weather.merge(stations, on='station', how='outer', indicator=True)
  cross_join = weather.merge(stations, how='cross') # carthesian product

  assert get_row_count(
    inner_join, left_join, 
    right_join, outer_join, 
    cross_join, 
  ) == [
    78780, 78949, 
    78949, 78949, 
    21979620
  ]

  dirty_data = pd.read_csv(
    'data/dirty_data.csv', index_col='date'
  ).drop_duplicates().drop(columns='SNWD')

  valid_station = dirty_data.query('station != "?"').drop(columns=['WESF', 'station'])
  station_with_wesf = dirty_data.query('station == "?"').drop(columns=['station', 'TOBS', 'TMIN', 'TMAX'])

  merged = valid_station.merge(
    station_with_wesf, 
    how='left',
    left_index=True,  # the column to use from the left dataframe is the index
    right_index=True,  # the column to use from the right dataframe is also the index
    suffixes=('', '_?')
  ).query('WESF > 0')

  # joining on the index is more efficient by using the join() method
  joined = valid_station.join(
    station_with_wesf, 
    how='left', 
    lsuffix='', 
    rsuffix='_?'
  ).query('WESF > 0')

  assert merged.equals(joined)

  # using the set operation to join the two dataframes
  weather.set_index('station', inplace=True)
  stations.set_index('station', inplace=True)

  # explore what will happen with innner join by interseting 
  # the two indices to show overlapping stations
  assert weather.index.intersection(stations.index).nunique() == 110

  # number of stations to lose in the inner join from the
  # weather DataFrame
  weather.index.difference(stations.index).nunique() == 0

  # number of stations to lose in the inner join from the
  # stations DataFrame
  stations.index.difference(weather.index).nunique() == 169

  # alternatively, using symmetric difference to find the stations 
  # to lose gives the same result
  weather.index.symmetric_difference(stations.index).nunique() == 169

def test_data_enrichment():

  breakpoint()
  pass



if __name__ == "__main__":
  test_data_aggregation()