# For data loading
import scipy.io
# For manipulation of the arrays
import numpy as np
# For file manipulation and locating
import os
# For plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy

DATA_DIR = "data/"
EFFICIENT = True

def sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    p = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues_ = copy.deepcopy(eigenvalues[p])
    eigenvectors_ = copy.deepcopy(np.real(eigenvectors[:,p]))
    return eigenvalues_, eigenvectors_


def load_mat(file_path, features = "X", labels = "l"):
    """
    Loading of the data

    Parameters
    ----------
    file_path: str
        Name of the `.mat` input file
    features: str
        Name of the sheet for the features in the '.mat' input file
    labels: str
        Name of the sheet of the labels in the '.mat' input file

    Returns
    -------
    data: tuple
        * X: data
        * Y: labels
    """
    data = scipy.io.loadmat(file_path)
    X = data[features]
    Y = data[labels]

    assert(X.shape[1] == Y.shape[1])

    return (X, Y)

def remove_mean(data, mean = None):
    """
    Removes the mean of the matrix

    Parameters
    ----------
    data: numpy matrix
        Data matrix with features
    mean: numpy matrix
        If mean is none, use the mean from the data

    Returns
    -------
    data: numpy matrix
    mean: mean of the data
    """
    A = np.matrix(data)
    if mean is None:
        mean = A.mean(axis=1).reshape(-1, 1)
        return A - mean, mean
    else:
        return A - mean

def separate_data(data, train = 0.8):
    """
    Separates the data into train and test portion

    Parameters
    ----------
    data_in: tuple
        X : features
        y : labels

    Returns
    -------
    data: list
        X_train, y_train : train data
        X_val, y_val : validation data
    mean: numpy array
    """
    X,[y] = data
    X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2,stratify=y)

    # Adjust data orientation
    X_train = X_train.T
    X_test  = X_test.T

    data = [[X_train, y_train], [X_test, y_test]]

    '''
    np.random.seed(13)

    D, N = X.shape

    # Cardinality of labels
    card = len(set(y.ravel()))
    step = int(N / card)
    bounds = np.arange(0, N, step)

    shapes = list(zip(bounds[:-1], bounds[1:]))

    # Training Mask
    training_mask = []

    for shape in shapes:
        idx = np.random.choice(
            np.arange(*shape), int(train * step), replace=False)
        training_mask.append(idx)

    mask_train = np.array(training_mask).ravel()

    mask_test = np.array(list(set(np.arange(0, N)) - set(mask_train)))

    # Partition dataset to train and test sets
    X_train, X_test = X[:, mask_train], X[:, mask_test]
    y_train, y_test = y[:, mask_train], y[:, mask_test]

    data = [[X_train, y_train], [X_test, y_test]]

    '''
    return data

def compute_covariance(data):
    """
    Separates the data into train and test portion

    Parameters
    ----------
    data: numpy matrix

    Returns
    -------
    covariance matrix: numpy matrix

    """
    D, N = data.shape

    return (1/N) * np.dot(data,data.transpose())

def compute_eigenvalues_eigenvectors(data):
    """
    Computes the eigenvalues and eigenvectors of the data

    Parameters
    ----------
    data: numpy matrix

    Returns
    -------
    eigenvalues: numpy matrix
    eigenvectors: numpy matrix
    """
    return np.linalg.eig(data)

def compute_eigenspace(data):
    # Obtain covariance matrix from transform
    D,N = data.shape
    cov = (1/N) * np.dot(data.T, data)
    # Compute Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(cov)
    # Transform Eigenvectors to the Face plane
    eigenvectors = copy.deepcopy(np.matmul(data,eigenvectors))
    # Normalise Eigenvectors
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors,axis=0)

    # Sort eigenvalues and eigenvectors
    return sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors)


def preprocess():
    """
    1. Preprocesses the dataset into two portions
    2. Removes mean computed from the training dataset
    from the training dataset and the same mean is removed from
    the test dataset
    3. Displays his mean face for verification
    4. Saves the results for caching

    Parameters
    ----------
    None

    Returns
    -------
    S: list
        * Covariance matrices for training and test data
    Eigenvectors: list
        * Eigenvectors for training and test data
    Eigenvalues: list
        * Eigenvalues for training and test data
    data: list
        * Pre-processed data with already removed means together with labels

    """
    # Load the data from the matrix
    X, Y = load_mat(DATA_DIR + "face.mat")
    dataset = separate_data((X,Y))

    # Compute and remove mean from the training dataset
    dataset[0][0], mean = remove_mean(dataset[0][0])

    # Remove the same mean from the test dataset
    dataset[1][0] = remove_mean(dataset[1][0], mean)

    # Prepare the containers
    types = ["training", "test"]
    S = []
    Eigenvectors = []
    Eigenvalues = []

    for data in dataset:
        # Remove the last row that represents labels
        face_matrix = copy.deepcopy(data[0])

        cov = None
        eigenvalues  = None
        eigenvectors = None
        if EFFICIENT:
            # Obtain covariance matrix from transform
            D,N = face_matrix.shape
            A = copy.deepcopy(face_matrix)
            cov = (1/N) * np.dot(A.T, A)
            # Compute Eigenvalues and Eigenvectors
            eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(cov)
            # Transform Eigenvectors to the Face plane
            eigenvectors = copy.deepcopy(np.matmul(A,eigenvectors))
            # Normalise Eigenvectors
            eigenvectors = eigenvectors / np.linalg.norm(eigenvectors,axis=0)
        else:
            A = copy.deepcopy(face_matrix)
            cov = compute_covariance(A)
            eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(cov)

        # Sort eigenvalues and eigenvectors
        eigenvalues, eigenvectors = sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors)

        # Append to the container
        Eigenvectors.append(eigenvectors)
        Eigenvalues.append(eigenvalues)
        S.append(cov)
    i = 0

    # Save the data so that you do not have to do this over and over again
    for t in types:
        np.save(DATA_DIR +"processed/covariance_matrices/" + "{}.npy".format(t), S[i])
        np.save(DATA_DIR +"processed/eigenvectors/" + "{}.npy".format(t), Eigenvectors[i])
        np.save(DATA_DIR +"processed/eigenvalues/" + "{}.npy".format(t), Eigenvalues[i])
        np.save(DATA_DIR +"processed/data/" + "{}.npy".format(t),dataset[i][0])
        np.save(DATA_DIR +"processed/labels/" + "{}.npy".format(t),dataset[i][1])
        i+=1
    np.save(DATA_DIR+"processed/mean.npy",mean)
    return mean, Eigenvectors, Eigenvalues, dataset

def load_data():
    """
    Load the cached data or call preprocess()
    to generate new data

    Parameters
    ----------
    None

    Returns
    -------
    S: list
        * Covariance matrices for training and test data
    Eigenvectors: list
        * Eigenvectors for training and test data
    Eigenvalues: list
        * Eigenvalues for training and test data
    data: list
        * Pre-processed data with already removed means together with labels

    """
    if not os.path.exists(os.path.join(DATA_DIR, "processed/covariance_matrices/", "training.npy")):
        return preprocess()
    types = ["training", "test"]
    S = []
    Eigenvectors = []
    Eigenvalues = []
    dataset = []
    for t in types:
        S.append(np.load(DATA_DIR +"processed/covariance_matrices/" + "{}.npy".format(t)))
        Eigenvectors.append(np.load(DATA_DIR +"processed/eigenvectors/" + "{}.npy".format(t)))
        Eigenvalues.append(np.load(DATA_DIR +"processed/eigenvalues/" + "{}.npy".format(t)))
        X = np.load(DATA_DIR +"processed/data/" + "{}.npy".format(t))
        Y = np.load(DATA_DIR +"processed/labels/" + "{}.npy".format(t))
        dataset.append([X,Y])
    mean = np.load(DATA_DIR+"processed/mean.npy")
    return mean, Eigenvectors, Eigenvalues, dataset

if __name__ == '__main__':
    preprocess()
