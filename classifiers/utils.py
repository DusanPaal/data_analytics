'''Provides utility functions for the binary_classifier package.'''

# built-in imports
from collections import defaultdict

# third-party imports
import numpy as np

def random_classifier(test_labels) -> float:
    '''Calculates the accuracy of a random 
    classifier as a baseline for comparison
    with trained classifiers.

    The accuracy is calculated as the proportion 
    of correctly classified labels and represents
    the probability of randomly guessing the correct
    label.

    Parameters:
    -----------
    test_labels : list
        A list of labels to classify.

    Returns:
    --------
    float
        The accuracy of a random classifier.
    '''
    
    test_labels_cpy = test_labels.copy()
    np.random.shuffle(test_labels_cpy)
    hits_array = np.array(test_labels) == np.array(test_labels_cpy)
    p = np.sum(hits_array) / len(test_labels)

    return float(p)

def normalize(train_data: np.array, test_data: np.array) -> np.array:
    '''Normalizes the data by subtracting the mean
    and dividing by the standard deviation.

    Parameters:
    -----------
    train_data : np.array
    The training data to normalize.
    
    test_data : np.array
    The test data to normalize.

    Returns:
    --------
    tuple
    A tuple containing the normalized training
    and test data.
    '''

    train_mean = np.mean(train_data, axis=0)
    train_sd = np.std(train_data, axis=0)

    train_data_norm = (train_data - train_mean) / train_sd

    # to normalize the test data, we use the mean 
    # and standard deviation of the training data 
    # to avoid data leakage from the test set to 
    # the training set
    test_data_norm = (test_data - train_mean) / train_sd

    return (train_data_norm, test_data_norm)

def print_performance_metrics(
        metrics: dict, newline=True,
        n_decimals:int=2, direction: str = 'column'
    ) -> None:
    '''Prints the performance metrics of a model.
    
    Parameters:
    -----------
    metrics : dict
    A dictionary containing the evaluation metrics to print.
    
    newline : bool
    If True, prints a newline character after the metrics.

    n_decimals : int
    The number of decimal places to which the metrics values
    will be rounded (default is 2).

    direction : str
    The direction in which to print the metrics. If 'column',
    the metrics will be printed in a column format. If 'row',
    the metrics will be printed in a row format.
    '''

    if direction == 'column':
        endchar='\n'
    elif direction == 'row':
        endchar=' '
    else:
        raise ValueError(
            'Direction must be either "column" or "row"!'
        )

    for key, val in metrics.items():

        if isinstance(val, float):
            val = round(val, n_decimals)
        elif isinstance(val, list):
            val = [round(v, n_decimals) for v in val]

        print(f'{key}: {val}', end=endchar)

    if newline:
        print('\n')

def smooth_curve(points: list, factor: float=0.9) -> list:
    '''Smooths a curve by applying an
    exponential moving average.

    Parameters:
    -----------
    points : list
    A list of points to smooth.

    factor : float
    The smoothing factor (default is 0.9).

    Returns:
    --------
    list:
        A list of smoothed points
    '''

    smoothed_points = []

    for pt in points:
        if smoothed_points:
            prev = smoothed_points[-1]
            smoothed_points.append(prev * factor + pt * (1-factor))
        else:
            smoothed_points.append(pt)

    return smoothed_points


def find_duplicated_files(paths: list) -> list:
    '''Identifies and returns a list of 
    duplicated files based on their content.

    Parameters:
    -----------
    paths : list
    A list of file paths to check for duplicates.

    Returns:
    --------
    list:
        A list of lists, where each inner list
        contains the paths of files that are 
        duplicates of each other.
    '''
    
    hashes = defaultdict(list)
    duplicated = []
    
    for s in paths:
        with open(s, 'rb') as f:
            data = f.read()
        h = hash(data)
        hashes[h].append(s)
    
    for h, files in hashes.items():
        if len(files) > 1:
            record = []
            for f in files:
                record.append(f)
            duplicated.append(record)

    return duplicated
