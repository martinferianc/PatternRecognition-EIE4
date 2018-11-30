import numpy as np
import os
import copy
from  sklearn.metrics.pairwise import pairwise_kernels

def sort_eigenvalues_eigenvectors(eigenvectors, eigenvalues):
    p = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues_ = copy.deepcopy(eigenvalues[p])
    eigenvectors_ = copy.deepcopy(np.real(eigenvectors[:,p]))
    return eigenvectors_, eigenvalues_

class LDA:
    def __init__(self, metric = "rbf"):
        self.fitted = False
        self.K = None
        self.V = None
        self.V_w = None
        self.iD_w = None
        self.M = None
        self.M_N = None
        self.I = None
        self.X = None
        self.n = None
        self.N = None
        self.F = None
        self.C = None
        self.metric = metric
        self.W_pseudo = None
        self.X = None


    def fit(self, X, labels, refit=False):

        self.X = X
        self.F,self.N = self.X.shape
        # Compute K
        K = None
        if os.path.exists(os.path.join("weights/K.npy")) and refit is False:
            K = np.load("weights/K.npy")
        else:
            K = pairwise_kernels(self.X.T, metric=self.metric)
            np.save("weights/K.npy", K)
        classes = np.unique(labels)
        self.C = len(classes)

        if os.path.exists(os.path.join("weights/E.npy")) and refit is False:
            self.E = np.load("weights/E.npy")
        else:
            self.E = np.zeros((self.N,self.N))
            offset = 0
            for i in range(self.C):
                N_ci = np.count_nonzero(classes[i] == labels)
                for j in range(N_ci):
                    for k in range(N_ci):
                        self.E[offset+j,offset+k] = (1/N_ci)
                offset+=N_ci
            np.save("weights/E.npy", self.E)

        if os.path.exists(os.path.join("weights/M.npy")) and refit is False:
            self.M = np.load("weights/M.npy")
            self.I = np.load("weights/I.npy")
        else:
            self.I = np.identity(self.N)
            self.M = (self.I-self.E)
            np.save("weights/I.npy", self.I)
            np.save("weights/M.npy", self.M)

        if os.path.exists(os.path.join("weights/M_N.npy")) and refit is False:
            self.M_N = np.load("weights/M_N.npy")
        else:
            ones = np.ones(self.N)
            self.M_N = self.I - (1/self.N)*np.matmul(ones,ones.T)
            np.save("weights/M_N.npy", self.M_N)

        print("Eigendecomposition of K_w...")
        if os.path.exists(os.path.join("weights/D_w.npy")) and refit is False:
            self.K_w = np.load("weights/K_w.npy")
            self.D_w = np.load("weights/D_w.npy")
            self.iD_w = np.load("weights/iD_w.npy")
            self.V_w = np.load("weights/V_w.npy")
        else:
            self.K_w = np.matmul(np.matmul(self.M,K),self.M)
            self.K_w = (self.K_w + self.K_w.T)*0.5
            self.D_w,self.V_w = np.linalg.eig(self.K_w)
            self.D_w[np.abs(self.D_w)<1e-10] = 0.0
            self.D_w = np.real(self.D_w)
            self.iD_w = copy.deepcopy(self.D_w)
            for i in range(len(self.D_w)):
                if self.iD_w[i]!=0.0:
                    self.iD_w[i] = self.iD_w[i]**(-1)
            self.iD_w = np.diag(self.iD_w)
            np.save("weights/iD_w.npy", self.iD_w)
            np.save("weights/D_w.npy", self.D_w)
            np.save("weights/V_w.npy", self.V_w)
            np.save("weights/K_w.npy", self.K_w)
        print("Eigendecomposition of S_b...")
        T = None
        if os.path.exists(os.path.join("weights/T.npy")) and refit is False:
            T = np.load("weights/T.npy")
            self.D = np.load("weights/D.npy")
            self.V = np.load("weights/V.npy")
            self.S_b = np.load("weights/S_b.npy")
        else:
            T = np.matmul(np.matmul(np.matmul(self.iD_w, self.V_w.T), self.M), K)
            self.S_b = np.matmul(np.matmul(T,self.E),T.T)
            _D,_V = np.linalg.eig(self.S_b)
            _D[np.abs(_D)<1e-10] = 0.0
            _D = np.real(_D)
            _V, self.D = self._sort_eigenvalues_eigenvectors( _V, _D)
            self.V = _V[:,:self.C-1]
            np.save("weights/T.npy", T)
            np.save("weights/S_b.npy", self.S_b)
            np.save("weights/D.npy", self.D)
            np.save("weights/V.npy", self.V)

        if os.path.exists(os.path.join("weights/W_pseudo.npy")) and refit is False:
            self.W_pseudo = np.load("weights/W_pseudo.npy")
        else:
            self.W_pseudo = np.matmul(np.matmul(np.matmul(np.matmul(self.V.T, self.iD_w),self.V_w.T),self.M),self.M_N)
            np.save("weights/W_pseudo.npy", self.W_pseudo)

        self.fitted = True
        return True

    def transform(self, Y):
        if self.fitted is False:
            raise Exception("You have not fitted or loaded the data yet!")
        new_Y = []
        for y in Y.T:
            K = pairwise_kernels(self.X.T, y.reshape(1, -1), metric=self.metric)
            new_Y.append(np.abs(np.matmul(self.W_pseudo,K)).flatten())
        new_Y = np.array(new_Y)
        return np.array(new_Y)


    def _sort_eigenvalues_eigenvectors(self, eigenvectors, eigenvalues):
        p = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues_ = copy.deepcopy(eigenvalues[p])
        eigenvectors_ = copy.deepcopy(np.real(eigenvectors[:,p]))
        return eigenvectors_, eigenvalues_
