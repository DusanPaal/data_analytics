import itertools
import seaborn as sns
import matplotlib.pyplot as plt


def show_plot(title='', suptitle='', ticks=None, labels=None) -> None:
  """Displays a plot with the given title.

  Parameters:
  -----------
  title: str
    The title of the plot. 
    By default, no title is displayed.
          
  suptitle: str
    The super title of the plot.
    By default, no super title is displayed.

  ticks: ArrayLike or None
    The ticks to display on the x-axis.

  labels: ArrayLike or None
    The labels to display on the x-axis.
  """

  if title != '':
    plt.title(title)

  if suptitle != '':
    plt.suptitle(suptitle)

  if not(ticks is None or labels is None):
    plt.xticks(ticks, labels)
    
  plt.show()
    

def plot_regression_with_residues(data) -> list:
  """Plots regression plots and their corresponding residual 
  plots for all permutations of columns in the given DataFrame.

  Parameters:
  ----------
  data: pd.DataFrame
    The input data containing multiple columns for which 
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