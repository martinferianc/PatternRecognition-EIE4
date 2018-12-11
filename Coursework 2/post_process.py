# Import all the modules to determine the cofusion matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os


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
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("{}.png".format(file_path))
    plt.close()
