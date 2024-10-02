import numpy as np
import pandas as pd
from solver import print_matrix
import scipy

if __name__ == "__main__":

  # data = pd.read_csv(r"data/hrach.csv")
  # data = data.apply(lambda x: x.round(2))

  # # create covariance matrix from the data
  # cov_matrix = data.cov(numeric_only=True)
  # mask = np.triu(np.ones(cov_matrix.shape), 1).astype(np.bool)
  # cov_matrix[mask] = pd.NA
  # print_matrix(cov_matrix)

  # scipy.stats.pearsonr(data['Aro'], data['Slad'])

  a = np.matrix([
    [1, 2],
    [3, 4],
    [3, 4],
    [3, 4],
    [3, 4],
  ])

  b = np.array([68,85, 83, 65, 78])
  # map the array to ascii characters
  print("".join(map(chr, b)))

  

  # convert teh 'a' character to an integer
  print(ord('a'))

  print(data)