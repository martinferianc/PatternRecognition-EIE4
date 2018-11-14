# Import all the modules to determine the cofusion matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import os


def calculate_scores(y_test, y_pred, file_path):
    """
    Calculates precision, recall and F1 score with respect to the labels
    Stores into a file

    Parameters
    ----------
    y_test: list
        Already given labels
    y_pred:
        Predictions made by the model
    file_path: str
        Name of the of the file where the results should be stored,
        together with a path

    Returns
    -------
    A file with the respective scores
    """
    with open("{}.txt".format(file_path), "w") as text_file:

        # It might happen that a quality can be not represented at all in a test, training, val set
        # Hence check for that case because the classifier might not have seen that quality at all
        classes_pred = [str(i) for i in np.unique(y_pred)]
        classes_test = [str(i) for i in np.unique(y_test)]

        classes = None
        if len(classes_pred)>len(classes_test):
            classes = classes_pred
        else:
            classes = classes_test

        p_score =  precision_score(y_test, y_pred, average=None)
        text_file.write("Labels:\n{}\n".format(classes))
        text_file.write("Precision scores:\n{}\n".format(p_score))

        r_score =  recall_score(y_test, y_pred, average=None)
        text_file.write("Labels:\n{}\n".format(classes))
        text_file.write("Recall scores:\n{}\n".format(r_score))

        f1 =  f1_score(y_test, y_pred, average=None)
        text_file.write("Labels:\n{}\n".format(classes))
        text_file.write("F1 scores:\n{}".format(f1))
