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
    X, [y] = load_mat('data/face.mat')
    dataset = {'train_x': [], 'train_y': [], 'test_x': [], 'test_y': []}

    # Perform train,test split
    dataset['train_x'], dataset['test_x'], dataset['train_y'], dataset['test_y'] = train_test_split(X.T, y, test_size=0.2,stratify=y)

    # Adjust data orientation
    dataset['train_x'] = dataset['train_x'].T
    dataset['test_x']  = dataset['test_x'].T

    types = ["training", "test"]

    # Save the data so that you do not have to do this over and over again
    i = 0

    np.save(DATA_DIR +"processed_raw/data/training.npy",dataset['train_x'])
    np.save(DATA_DIR +"processed_raw/labels/training.npy",dataset['train_y'])

    np.save(DATA_DIR +"processed_raw/data/test.npy",dataset['test_x'])
    np.save(DATA_DIR +"processed_raw/labels/test.npy",dataset['test_y'])

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
