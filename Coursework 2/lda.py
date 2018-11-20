import numpy as np
import os
import copy

def sort_eigenvalues_eigenvectors(eigenvectors, eigenvalues):
    p = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues_ = copy.deepcopy(eigenvalues[p])
    eigenvectors_ = copy.deepcopy(np.real(eigenvectors[:,p]))
    return eigenvectors_, eigenvalues_

def PCA(_X, n):
    X = _X - np.mean(_X,1,keepdims=True);

    # Calculate the covariance matrix
    c = np.matmul(X.T,X)
    c = 0.5*(c + c.T)

    # Compute the eigenvalues and eigenvectors
    D,V = np.linalg.eig(c)
    iD = D**(-1/2)

    # Convert to original eigenvectors
    U = np.matmul(X,np.matmul(V,np.diag(iD)))

    U,D = sort_eigenvalues_eigenvectors(U,D)

    return U[:,:n]


def wPCA(_X, n):

    X = _X - np.mean(_X,1,keepdims=True)

    # Calculate the covariance matrix
    c = np.matmul(X.T,X)
    c = 0.5*(c + c.T)

    # Compute the eigenvalues and eigenvectors
    D,V = np.linalg.eig(c)
    iD = D**(-1)

    # Convert to original eigenvectors
    U = np.matmul(X,np.matmul(V,np.diag(iD)))

    U,D = sort_eigenvalues_eigenvectors(U,D)

    return U[:,:n]


def LDA(X, labels):
    if os.path.exists(os.path.join("w.npy")):
        return np.load("w.npy")

    classes = np.unique(labels)

    c = len(classes)

    F,N = X.shape

    M = np.zeros((N,N))
    offset = 0
    for i in range(c):
        N_ci = np.count_nonzero(classes[i] == labels)
        for j in range(N_ci):
            for k in range(N_ci):
                M[offset+j,offset+k] = (1/N_ci)
        offset+=N_ci

    # Created M
    print("Created M...")

    I = np.identity(N)
    X_W = np.matmul(X,(I-M))

    print("Performing wPCA...")
    U = wPCA(X_W, N-c)

    print("Performing PCA...")
    X_B_tilde = np.matmul(np.matmul(U.T,X),M)
    Q = PCA(X_B_tilde, c-1)

    W = np.matmul(U,Q)
    np.save("w.npy", W)
    return W

def LDA_transform(W,X):
    return np.matmul(X.T,W)
