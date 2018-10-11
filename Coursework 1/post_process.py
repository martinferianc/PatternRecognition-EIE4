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
        text_file.write("Recall scores:\n{}".format(r_score))

        f1 =  f1_score(y_test, y_pred, average=None)
        text_file.write("Labels:\n{}\n".format(classes))
        text_file.write("F1 scores:\n{}".format(f1))

# This function calculates the confusion matrix and visualizes it
def plot_confusion_matrix(y_test, y_pred, file_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Calculates and plots a confusion matrix from
    the given labels

    Parameters
    ----------
    y_test: list
        Already given labels
    y_pred:
        Predictions made by the model
    file_path: str
        Name of the of the file where the results should be stored,
        together with a path
    nomralize: bool
        Whether the confusion matrix should ne bormalized
    title: str
        Whether the plot should have any special title
    cmap: plt.cm.*
        What cholor scheme should be used for plotting

    Returns
    -------
    An image of confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    classes_pred = [str(i) for i in np.unique(y_pred)]
    classes_test = [str(i) for i in np.unique(y_test)]

    classes = None
    if len(classes_pred)>len(classes_test):
        classes = classes_pred
    else:
        classes = classes_test

    # In case the confusion matrix should be normalized
    if normalize:
        t = cm.sum(axis=1)[:, np.newaxis]
        for i in t:
            if i[0] == 0:
                i[0] = 1
        cm = cm.astype('float') / t

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.around(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig("{}.png".format(file_path))

    plt.close()
