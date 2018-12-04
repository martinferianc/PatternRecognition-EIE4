import numpy as np


import numpy as np
# For file manipulation and locating
import os
# For plotting
from tqdm import tqdm

import copy

from pre_process import load_data as ld
from sklearn.preprocessing import normalize


# We define some constants that we are going to reuse
DATA_DIR = "data/"
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"

def save_data(data, file_path, name):
    """
    Saves the data
    given the name and
    the file path

    Parameters
    ----------
    data: numpy matrix
        Data matrix with features
    file_path: str
        File path where the file should be saved
    name: str
        Specific name of the given file

    """
    np.save(file_path + "{}.npy".format(name),data)

def preprocess(size = 100000,lower_bound=0, upper_bound = 7368,samples = 10, same_class=0.4, different = 0.5, penalty = 10):
    """
    1. Preprocesses the dataset into three splits: training, validation, test
    2. Performs z normalization on the three different chunks
    3. Saves the data

    Parameters
    ----------
    None

    Returns
    -------
    all_data: list
        * All the data split into lists of [features, labels]

    """
    all_data = ld(False)
    training_data = all_data[0]

    training_labels = training_data[1]
    training_features = training_data[0]

    training_features = normalize(training_features, axis=1)

    training_N,F = training_features.shape

    X = []
    Y = []
    values = []

    C = int(samples*same_class)
    D = int(samples*different)

    for i in tqdm(range(int(size/samples))):
        random_i = np.random.randint(lower_bound,upper_bound)
        C_counter = 0
        D_counter = 0
        offset = 0
        while D_counter<D:
            random_j = np.random.randint(lower_bound,upper_bound)
            if training_labels[random_i] != training_labels[random_j]:
                D_counter+=1
                offset+=1
                X.append(copy.deepcopy(training_features[random_i]))
                Y.append(copy.deepcopy(training_features[random_j]))
                values.append(penalty)

        while C_counter<C:
            low = 0
            high = training_N
            if random_i-10>lower_bound:
                low = random_i-10
            if random_i+10<upper_bound:
                high = random_i+10
            random_j = np.random.randint(lower_bound, upper_bound)
            if training_labels[random_i] == training_labels[random_j] and random_i!=random_j:
                C_counter+=1
                offset +=1
                X.append(copy.deepcopy(training_features[random_i]))
                Y.append(copy.deepcopy(training_features[random_j]))
                values.append(1)

        while offset < samples:
            X.append(copy.deepcopy(training_features[random_i]))
            Y.append(copy.deepcopy(training_features[random_i]))
            offset+=1
            values.append(0)

    indeces = np.random.choice(size, size=size, replace=False)
    X = np.array(X)
    Y = np.array(Y)
    values = np.array(values)

    return [X[indeces], Y[indeces], values[indeces]]

def load_data(retrain=False):
    """
    Load the cached data or call preprocess()
    to generate new data

    Parameters
    ----------
    None

    Returns
    -------
    all_data: list
        * All the data split into lists of [features, labels]

    """
    if retrain is True:
        print("Generating new data...")
        X_train, Y_train, values_train = preprocess(50000, 0, 6368)
        X_validation, Y_validation, values_validation = preprocess(10000, 6369)
        save_data(X_train,PROCESSED_DIR,"training_nn_X")
        save_data(Y_train,PROCESSED_DIR,"training_nn_Y")
        save_data(values_train,PROCESSED_DIR,"training_nn_values")
        save_data(X_validation,PROCESSED_DIR,"validation_nn_X")
        save_data(Y_validation,PROCESSED_DIR,"validation_nn_Y")
        save_data(values_validation,PROCESSED_DIR,"validation_nn_values")
        return [X_train, Y_train, values_train, X_validation, Y_validation, values_validation]
    else:
        print("Loading data...")
        data = []
        data.append(np.load(PROCESSED_DIR + "training_nn_X.npy"))
        data.append(np.load(PROCESSED_DIR + "training_nn_Y.npy"))
        data.append(np.load(PROCESSED_DIR + "training_nn_values.npy"))
        data.append(np.load(PROCESSED_DIR + "validation_nn_X.npy"))
        data.append(np.load(PROCESSED_DIR + "validation_nn_Y.npy"))
        data.append(np.load(PROCESSED_DIR + "validation_nn_values.npy"))
        return data

if __name__ == '__main__':
    load_data(retrain=True)
