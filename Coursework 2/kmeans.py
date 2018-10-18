# For loading the mat data
from pre_process import load_data
# For calculatuons
import numpy as np
# Import the KMeans library
from sklearn.cluster import KMeans
# Import matplotlib
import matplotlib.pyplot as plt
# Import post process analysing methods
from post_process import calculate_scores, plot_confusion_matrix

def test(data, type="naive"):
    """
    Evaluate different types
    of KMeans classfier

    Parameters
    ----------
    data: list
        Data which can be training, validation or
        test including the labels
    type: str
        Which implementation of kNN should we look
        k for
    Returns
    -------
    error: float
        * Error computed on the test set
    test_labels: list
        * Labels needed for the further analysis
    predicted_labels: list
        * Labels needed for the further analysis
    """

    training_data = data[0][0]
    training_labels = data[0][1]

    test_data = data[2][0]
    test_labels = data[2][1]

    error = 0
    print("Fitting the model...")
    clf = None
    if clf == "naive":
        clf = KMeans(n_clusters=len(set(training_labels)))
    else:
        raise Exception("Wrong method!")
    clf.fit(training_data, training_labels)

    # Predict
    predicted_labels = clf.predict(test_data)
    error = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]!=test_labels[i]:
            error+=1
    error/=len(validation_labels)
    print(error)

    return error, predicted_labels, test_labels

def analyse_KMeans():
    """
    Analyse and collect all the different results
    with respect to different KMeans tests
    """

    # Define all the different methods that we are going to try
    methods = ["naive"]
    for method in methods:
        all_data = load_data()
        error, test_labels, predicted_labels = test(all_data, t)
        calculate_scores(test_labels, predicted_labels, "results/KMeans/{}_scores".format(method))
        plot_confusion_matrix(test_labels, predicted_labels, "results/KMeans/{}_CM".format(method))
        plot_confusion_matrix(test_labels, predicted_labels, True, "results/KMeans/{}_CM_normalized".format(method))


if __name__ == '__main__':
    analyse_KMeans()
