import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read data from the file
# data = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
# https://opendata.ecdc.europa.eu/covid19/casedistribution/csv.

data = pd.read_csv('data/covid19/cases.csv')

data = data.reset_index()


pivotted = data.pivot(index='date', columns='countriesAndTerritories', values='cases')
pivotted.fillna(0, inplace=True)

breakpoint()
pass