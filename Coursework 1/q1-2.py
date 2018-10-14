import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm
import copy

from pre_process import load_mat, remove_mean, separate_data, compute_eigenvalues_eigenvectors
from eigenfaces import EigenFace

def main():
    # Load the data from the matrix
    X, Y = load_mat("data/face.mat")
    dataset = separate_data((X,Y))

    # Compute and remove mean from the training dataset
    mean = dataset[0][0].mean(axis=1).reshape(-1,1)

    dataset[0][0] = dataset[0][0] - mean
    dataset[1][0] = dataset[1][0] - mean

    train_faces = dataset[0][0]
    D, N = train_faces.shape

    #'''
    # Naive Eigenvectors
    S_naive = copy.deepcopy((1 / N) * np.dot(train_faces, train_faces.T))
    _l_naive,_u_naive = np.linalg.eig(S_naive)
    #indexes = np.argsort(np.abs(_l_naive))[::-1]
    u_naive = np.real(_u_naive)
    l_naive = _l_naive

    naive_eigenvectors = u_naive

    naive = EigenFace(dataset,naive_eigenvectors,mean)
    M = [5,50,100,200,500,750,1000,1500,2000]
    err = []
    for m in M:
        naive.M = m
        err.append(naive.run_reconstruction())
    plt.scatter(M,err)
    plt.show()
    '''


    # Efficient Eigenvectors
    S_efficient = copy.deepcopy((1/N) * np.dot(train_faces.T, train_faces))
    _l_efficient,_u_efficient = np.linalg.eig(S_efficient)
    _u_efficient = copy.deepcopy(np.dot(train_faces, copy.deepcopy(_u_efficient)))
    u_efficient = np.real(_u_efficient)
    l_efficient = _l_efficient

    efficient_eigenvectors = u_efficient

    efficient = EigenFace(dataset,efficient_eigenvectors,mean)
    efficient.run_reconstruction()
    '''

if __name__ == '__main__':
    main()
