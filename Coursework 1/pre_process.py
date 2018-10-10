import scipy.io
import numpy as np
import copy
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_DIR = "data/"

# Load the data and convert it into numpy matrices
def load_mat(file_path, features = "X", labels = "l"):
    data = scipy.io.loadmat(file_path)
    X = data[features]
    Y = data[labels]
    return (X, Y)

# Remove the overall mean from the data
def remove_mean(data):
    A = np.matrix(data)
    mean = A.mean(axis=1)
    return A - mean, mean

# Separate the data into training, validation and test sets
# Reshuffle the data
def separate_data(data_in):
    X,y = data_in

    # Get the mean image
    X, mean = remove_mean(X)
    X = X.transpose()
    y = y.transpose()

    # Train size is 80%
    # Validation size is 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    data = [[X_train, y_train], [X_test, y_test]]
    for d in data:
        d[0] = d[0].transpose()
        d[1] = d[1].transpose()
    return data, mean

def compute_covariance(data):
    return np.cov(data)

def compute_eigenvalues_eigenvectors(data):
    return np.linalg.eig(data)

def display_mean(mean):
    mean = mean.reshape((46,56))
    mean = np.rot90(mean,3)
    plt.imshow(mean, cmap="gray")
    plt.show()

def main():
    # Load the data from the matrix
    X, Y = load_mat(DATA_DIR + "face.mat")
    dataset, mean = separate_data((X,Y))

    # Display the mean face for verification
    display_mean(mean)

    # Prepare the containers
    types = ["training", "test"]
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
        i+=1
    return S, Eigenvectors, Eigenvalues, dataset

# API to load any numpy data
def load_data():
    if not os.path.exists(os.path.join(DATA_DIR, "processed/covariance_matrices/", "training.npy")):
        return main()
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
        print(X.shape)
        Y = np.load(DATA_DIR +"processed/labels/" + "{}.npy".format(t))
        dataset.append([X,Y])
    return S, Eigenvectors, Eigenvalues, dataset

if __name__ == '__main__':
    main()
