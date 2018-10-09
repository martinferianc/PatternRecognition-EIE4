import scipy.io
import numpy as np
import copy
import os
from sklearn.model_selection import train_test_split

DATA_DIR = "data/"

# Load the data and convert it into numpy matrices
def load_mat(file_path, features = "X", labels = "l"):
    data = scipy.io.loadmat(file_path)
    X = data[features]
    Y = data[labels]
    return (X, Y)

# API to load any numpy data
def load_data(file_path):
    data = np.load(file_path)
    return data

# Remove the overall mean from the data
def remove_mean(data):
    A = np.matrix(data)
    mean = A.mean(axis=1)
    return A - mean

# Separate the data into training, validation and test sets
# Reshuffle the data
def separate_data(data_in):
    X,y = data_in
    # Train size is 80%
    # Validation size is 10%
    # Test size is 10%
    X = X.transpose()
    y = y.transpose()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
    data = [[X_train, y_train], [X_val, y_val], [X_test, y_test]]
    for d in data:
        d[0] = d[0].transpose()
        d[1] = d[1].transpose()
    return data

def compute_covariance(data):
    return np.cov(data)

def compute_eigenvalues_eigenvectors(data):
    return np.linalg.eig(data)

def main():
    # Load the data from the matrix
    X, Y = load_mat(DATA_DIR + "face.mat")
    dataset = separate_data((X,Y))

    # Prepare the containers
    types = ["training", "validation", "test"]
    S = []
    Eigenvectors = []
    Eigenvalues = []

    for data in dataset:
        # Remove the last row that represents labels
        data_without_labels = data[0]

        # Compute the covariance matrix
        cov = compute_covariance(data_without_labels)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(cov)

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
        print(S[i].shape)
        i+=1

if __name__ == '__main__':
    main()
