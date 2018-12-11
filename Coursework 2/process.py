import numpy as np

def weight(x):
    """
    Implement Gaussian  RBF

    Parameters
    ----------
    k: float
        Input distance

    Returns
    -------
    y: float
        Output transformation
    """
    return np.exp(-(x** 2))

def vote(labels, weights):
    """
    Iterate through all the votes and compare them
    based on the distance and return the label
    with the biggest score

    Parameters
    ----------
    x: numpy array
        Array of labels

    Returns
    -------
    y: float
        Output transformation
    """
    label = -1
    best_score = -float('inf')
    for i in range(len(labels)):
        score = 0
        for j in range(len(labels)):
            if labels[i] == labels[j]:
                score+=weights[j]
        if score> best_score:
            label = labels[i]
            best_score = score
    return label
