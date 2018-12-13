# For data loading
from scipy.io import loadmat
# For splitting the data into test, train, validation splits
from sklearn.model_selection import train_test_split
# For manipulation of the arrays
import numpy as np
# For file manipulation and locating
import os
# For plotting
import json
# For showing progress
from tqdm import tqdm

import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# We define some constants that we are going to reuse
DATA_DIR = "data/"
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"
N = 14096
TOTAL_SIZE = 2048

def plot_correlation_matrix(data, name):
    """
    Plot the correlation matrix for the features

    Parameters
    ----------
    data: numpy array
        Feature array
    name: string
        File name of the correlation matrix


    Returns
    -------
    """
    N,F = data.shape
    indeces = np.random.choice(N, size=100, replace=False)
    data = data[indeces,:]
    sns.set(style="white")

    d = pd.DataFrame(data=data)

    # Compute the correlation matrix
    corr = d.corr()

    fig, ax = plt.subplots(figsize=(100,100))
    cax = plt.matshow(corr, interpolation="nearest")
    plt.colorbar(cax)

    plt.title("Features",fontsize=12,y=1.08)
    plt.xlabel("Correlation matrix",  fontsize=12)
    plt.ylabel("Features",fontsize=12)
    plt.savefig("results/{}.png".format(name))

    plt.close()

def select_features(gallery_camIds, query_camId, gallery_labels, query_label, gallery_features):
    """
    Preselects features with the respective query

    Parameters
    ----------
    gallery_camIds: numpy array
        Camera IDs for the respective gallery images
    query_camId: int
        Id with respect to which we need to filter the dataset
    gallery_labels: numpy array
        Labels for the respective gallery images
    query_label: int
        label with respect to which we need to filter the dataset

    gallery_features: numpy array
        The gallery samples that we need to filter for this particular query

    Returns
    -------
    selected_gallery_samples: list
        *   pre-selected gallery samples

    selected_gallery_labels: list
        *   pre-selected gallery labels corresponding to each sample
    """
    selected_gallery_samples = []
    selected_gallery_labels = []
    for j in range(len(gallery_features)):
        if not (gallery_camIds[j]==query_camId and gallery_labels[j]==query_label):
            selected_gallery_samples.append(gallery_features[j])
            selected_gallery_labels.append(gallery_labels[j])

    selected_gallery_samples = np.array(selected_gallery_samples)
    selected_gallery_labels = np.array(selected_gallery_labels)
    return selected_gallery_samples, selected_gallery_labels

def load_mat(file_path, label):
    """
    Loading of the data indexes of the images

    Parameters
    ----------
    file_path: str
        Name of the `.mat` input file
    label: str
        Name of the sheet for the indexes in the '.mat' input file

    Returns
    -------
    idxs: list
        * idxs corresponding to the given category
    """
    idxs = loadmat(file_path)[label].flatten()

    return (idxs)

def normalize(data):
    """
    Removes the mean of the image
    normalizses it between 0 and 1
    among all data poings

    Parameters
    ----------
    data: numpy matrix
        Data matrix with features

    Returns
    -------
    _data: numpy matrix
    """
    _data = []
    shape = data.shape
    for i in tqdm(range(len(data))):
         _data.append(copy.deepcopy((data[i] - data[i].mean(axis=0)) / data[i].std(axis=0)))
    _data = np.array(_data)
    _data = _data.reshape(shape)
    return _data

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

def preprocess():
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
    types = ["training","query", "gallery"]
    print("Loading of index data...")

    labels = load_mat(RAW_DIR + "cuhk03_new_protocol_config_labeled.mat", "labels")
    _training_indexes = loadmat(RAW_DIR + 'cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
    _query_indexes = loadmat(RAW_DIR + 'cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()
    _gallery_indexes = loadmat(RAW_DIR + 'cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
    camIds = loadmat(RAW_DIR + 'cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()

    training_indexes = np.array([i-1 for i in _training_indexes])
    query_indexes = np.array([i-1 for i in _query_indexes])
    gallery_indexes = np.array([i-1 for i in _gallery_indexes])

    training_labels = labels[training_indexes]
    query_labels = labels[query_indexes]
    gallery_labels = labels[gallery_indexes]

    training_camId = camIds[training_indexes]
    query_camId = camIds[query_indexes]
    gallery_camId = camIds[gallery_indexes]

    print("Loading of features...")
    with open(RAW_DIR + "feature_data.json", 'r') as data:
        features = np.array(json.load(data))
        features = features.reshape((N,TOTAL_SIZE))

        _training_data = features[training_indexes,:]
        _query_data = features[query_indexes,:]
        _gallery_data = features[gallery_indexes,:]

        print("Normalizing data...")
        training_data = copy.deepcopy(_training_data)
        query_data = copy.deepcopy(_query_data)
        gallery_data = copy.deepcopy(_gallery_data)

        plot_correlation_matrix(training_data,"training_corr_matrix")
        plot_correlation_matrix(query_data,"query_corr_matrix")
        plot_correlation_matrix(gallery_data,"gallery_corr_matrix")

        training_data_normalized = normalize(_training_data)
        query_data_normalized = normalize(_query_data)
        gallery_data_normalized = normalize(_gallery_data)


        print("Saving data...")
        all_data = [[training_data, training_data_normalized ,training_labels, training_camId], \
                    [query_data, query_data_normalized, query_labels, query_camId], \
                    [gallery_data, gallery_data_normalized ,gallery_labels, gallery_camId]]
        for i,t in enumerate(types):
            save_data(all_data[i][0],PROCESSED_DIR,"{}_features".format(t))
            save_data(all_data[i][1],PROCESSED_DIR,"{}_normalized_features".format(t))
            save_data(all_data[i][2],PROCESSED_DIR,"{}_labels".format(t))
            save_data(all_data[i][3],PROCESSED_DIR,"{}_camId".format(t))

        return all_data

def load_data(z_normalized = True):
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
    if not os.path.exists(os.path.join(DATA_DIR, "processed/", "training_normalized_features.npy")):
        print("Generating new data...")
        all_data = preprocess()
        if z_normalized:
            del all_data[0][0]
            del all_data[1][0]
            del all_data[2][0]
        else:
            del all_data[0][1]
            del all_data[1][1]
            del all_data[2][1]
    print("Loading data...")

    types = ["training","query", "gallery"]
    all_data = []
    for t in types:
        data = []
        if z_normalized:
            data.append(np.load(PROCESSED_DIR + "{}_normalized_features.npy".format(t)))
        else:
            data.append(np.load(PROCESSED_DIR + "{}_features.npy".format(t)))
        data.append(np.load(PROCESSED_DIR + "{}_labels.npy".format(t)))
        data.append(np.load(PROCESSED_DIR + "{}_camId.npy".format(t)))
        all_data.append(data)
    print("Finished loading data...")
    return all_data

if __name__ == '__main__':
    preprocess()
