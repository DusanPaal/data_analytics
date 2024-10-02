import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def show_plot(title: str = '', suptitle: str = '') -> None:
  """Displays a plot with the given title.

  Parameters:
  -----------
  title: The title of the plot. 
          By default, no title is displayed.
          
  suptitle: The super title of the plot.
            By default, no super title is displayed.
  """

  if title != '':
    plt.title(title)

  if suptitle != '':
    plt.suptitle(suptitle)
    
  plt.show()

def plot_regression_with_residues(data: pd.DataFrame) -> list:
  """Plots regression plots and their corresponding residual 
  plots for all permutations of columns in the given DataFrame.

  Parameters:
  ----------
  data: The input data containing multiple columns for which 
        regression and residual plots will be generated.

  Returns:
  --------
  A list of matplotlib.axes.Axes objects representing 
  the generated plots.
  """

  n_columns = data.shape[1]
  n_permutations = n_columns * (n_columns - 1)

  fig, ax = plt.subplots(n_permutations, 2, figsize=(15, 8))
  permutations = itertools.permutations(data.columns, 2)
  cycle = itertools.cycle(['royalblue', 'darkorange'])

  for (x, y), axes, color in zip(permutations, ax, cycle):
    for subplot, func in zip(axes, (sns.regplot, sns.residplot)):
      func(x=x, y=y, data=data, ax=subplot, color=color)
      if func == sns.residplot:
        subplot.set_ylabel('Residuals')
        
  return fig.axes
