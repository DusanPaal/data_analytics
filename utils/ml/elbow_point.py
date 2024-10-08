"""Elbow plot to find value for k to use with k-means clustering"""

import matplotlib.pyplot as plt

def elbow_point(
    data, pipeline, kmeans_step_name='kmeans',
    k_range=range(1, 11), ax=None
  ):
    """Plot the elbow point to find an appropriate k
    for k-means clustering.

    The elbow point method involves creating multiple
    models with many values of k and plotting each model's
    inertia (within-cluster sum of squares) versus the 
    number of clusters. Minimizing the sum of squared 
    distances from points to their cluster's center gives
    the estimated optimal k.

    Parameters:
    -----------
    data: 
    The features to use.

    pipeline:
    The scikit-learn pipeline with `KMeans`.

    kmeans_step_name:
    The name of the `KMeans` step in the pipeline.

    k_range:
    The values of `k` to try.

    ax:
    Matplotlib `Axes` to plot on.

    Returns:
    --------
    A matplotlib `Axes` object
    """
    scores = []
    for k in k_range:
        pipeline.named_steps[kmeans_step_name].n_clusters = k
        pipeline.fit(data)
        scores.append(pipeline.score(data) * -1)

    if not ax:
        fig, ax = plt.subplots()
    ax.plot(k_range, scores, 'bo-')
    ax.set_xlabel('k')
    ax.set_ylabel('inertias')
    ax.set_title('Elbow Point Plot')

    return ax