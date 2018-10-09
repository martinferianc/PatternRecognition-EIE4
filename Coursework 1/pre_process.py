import scipy.io
import numpy as np
import copy
import os

DATA_DIR = "data/"

# Load the data and convert it into numpy matrices
def load_mat(file_path):
    data = scipy.io.loadmat(file_path)
    X = data["X"]
    Y = data["l"]
    return (X, Y)

def remove_mean(data):
    A = np.matrix(data)
    mean = A.mean(axis=1)
    return A - mean


def separate_data(data_in, validation=0.1, test=0.1):
    # Load the target data into two vars
    X, Y = data_in

    # Append Y to the end of X

    X = remove_mean(X)
    data = np.vstack([X,Y])

    # Reshuffle the data
    data = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
    tseg = int(len(X) * test)
    vseg = int((len(X) - tseg) * validation)

    # Split the data into test and non-test data
    i = 0
    tsplit_start = (i * tseg) % len(data)
    tsplit_end = (i * tseg + tseg) % len(data)
    test_data = data[tsplit_start:tsplit_end, :]

    # Split remaining data into training and validation
    r_data = np.vstack((data[0:tsplit_start, :], data[tsplit_end:, :]))
    vsplit_start = (i * vseg) % len(r_data)
    vsplit_end = (i*vseg + vseg)  % len(r_data)
    validation_data = r_data[vsplit_start:vsplit_end, :]
    training_data = np.vstack((r_data[0:vsplit_start, :], r_data[vsplit_end:, :]))

    return (training_data, validation_data, test_data)

def compute_covariance(data):
    return np.cov(data)

def compute_eigenvalues_eigenvectors(data):
    return np.linalg.eig(data)



def main():
    types = ["training", "validation", "test"]
    X, Y = load_mat(DATA_DIR + "face.mat")
    training_data, validation_data, test_data = separate_data((X,Y))
    S = []
    Eigenvectors = []
    Eigenvalues = []
    dataset = [training_data, validation_data, test_data]
    for data in dataset:
        print(data.shape)
        data_without_labels = data[:-1,:]
        cov = compute_covariance(data_without_labels)
        S.append(cov)
        eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(cov)
        Eigenvectors.append(eigenvectors)
        Eigenvalues.append(eigenvalues)
    i = 0
    for t in types:
        np.save(DATA_DIR +"processed/covariance_matrices/" + "{}.npy".format(t), S[i])
        np.save(DATA_DIR +"processed/eigenvectors/" + "{}.npy".format(t), Eigenvectors[i])
        np.save(DATA_DIR +"processed/eigenvalues/" + "{}.npy".format(t), Eigenvalues[i])
        np.save(DATA_DIR +"processed/data/" + "{}.npy".format(t),dataset[i][:-1,:])
        np.save(DATA_DIR +"processed/labels/" + "{}.npy".format(t),dataset[i][-1,:])
        print(dataset[i][-1,:].shape)
        i+=1

if __name__ == '__main__':
    main()
