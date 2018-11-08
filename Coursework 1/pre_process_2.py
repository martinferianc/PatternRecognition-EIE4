import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import matrix_power
from pre_process import *
from eigenfaces import *
import copy
from pre_process import sort_eigenvalues_eigenvectors
from sklearn.model_selection import train_test_split

def save_dataset(VAL=False):
    #variable initialisation
    dataset = {
        'train_x' : [],
        'test_x'  : [],
        'val_x'   : [],
        'train_y' : [],
        'test_y'  : [],
        'val_y'   : [],
    }

    total_mean = []

    # load raw data

    X, [y] = load_mat('data/face.mat')

    # Perform train,test split
    dataset['train_x'], dataset['test_x'], dataset['train_y'], dataset['test_y'] = train_test_split(X.T, y, test_size=0.2,stratify=y)

    # Adjust data orientation
    dataset['train_x'] = dataset['train_x'].T
    dataset['test_x']  = dataset['test_x'].T

    if VAL:
        dataset['val_x'], dataset['test_x'], dataset['val_y'], dataset['test_y'] = train_test_split(dataset['test_x'].T, dataset['test_y'], test_size=0.5,stratify=y)
        dataset['val_x'] = dataset['val_x'].T
        dataset['test_x']  = dataset['test_x'].T

    # organise into seperate classes
    ## Get set of all labels
    labels = set(dataset['train_y'])

    train_class_data = [ [] for i in range(len(labels))]
    train_class_mean = [ [] for i in range(len(labels))]

    for label in labels:
        # Get indices for those labels
        indices = []
        for j in range(dataset['train_y'].shape[0]):
            indices.append(j) if (dataset['train_y'][j] == label) else 0

        # Select class from those indices
        train_class_data[label-1] = copy.deepcopy(dataset['train_x'][:,indices])

    # Get mean for each class
    for i in range(len(train_class_data)):
        train_class_mean[i] = copy.deepcopy(train_class_data[i]).mean(axis=1).reshape(-1,1)

    # Get total mean
    total_mean = copy.deepcopy(dataset['train_x']).mean(axis=1).reshape(-1,1)

    # Save to file

if __name__ == '__main__':
    save_dataset()
