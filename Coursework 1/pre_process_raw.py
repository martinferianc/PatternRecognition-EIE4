import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import matrix_power
from pre_process import *
from eigenfaces import *
import copy
from pre_process import load_mat, separate_data

def save_dataset():

    # Load the data from the matrix
    X, Y = load_mat(DATA_DIR + "face.mat")
    dataset = separate_data((X,Y))

    types = ["training", "test"]

    # Save the data so that you do not have to do this over and over again
    i = 0
    for t in types:
        np.save(DATA_DIR +"processed_raw/data/" + "{}.npy".format(t),dataset[i][0])
        np.save(DATA_DIR +"processed_raw/labels/" + "{}.npy".format(t),dataset[i][1])
        i+=1

    return dataset

def load_data():

    dataset = {'train_x': [], 'train_y': [], 'test_x': [], 'test_y': []}

    dataset['train_x'] = np.load(DATA_DIR +"processed_raw/data/training.npy")
    dataset['train_y'] = np.load(DATA_DIR +"processed_raw/labels/training.npy").T

    dataset['test_x'] = np.load(DATA_DIR +"processed_raw/data/test.npy")
    dataset['test_y'] = np.load(DATA_DIR +"processed_raw/labels/test.npy").T

    return dataset

if __name__ == '__main__':
    save_dataset()
