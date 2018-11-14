import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import matrix_power
import copy
import os

def sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    p = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues_ = copy.deepcopy(eigenvalues[p])
    eigenvectors_ = copy.deepcopy(np.real(eigenvectors[:,p]))
    return eigenvalues_, eigenvectors_

class LDA:
    def __init__(self):

        # Dataset
        # - Stores the raw dataset
        # - note, don't remove mean or anything from this data
        self.dataset = {
            'train_x': [],
            'train_y': []
        }

        # Training Data class seperation
        # { 'class 1' : [ images of class 1 ], etc ... }
        self.train_class_data = {}
        # Stores mean of each class
        # { 'class 1' : class 1 mean, 'class 2': class 2 mean, etc ... }
        self.train_class_mean = {}

        # Total mean
        # - mean of training set
        self.total_mean = []

        # Projection Matrices
        self.w_pca = [] # pca vector
        self.w_lda = [] # lda vector
        self.w_opt = [] # optimal vector which combines both

        # Model HyperParameters
        # NOTE: M_lda < M_pca and M_pca < N
        self.M_pca = 0
        self.M_lda = 0


    def run_setup(self):
        """
        Splits daaset between classes and gets their mean

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.split_classes()
        self.get_mean()

    def get_mean(self):
        """
        Gets both the class mean and the total mean

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Get Total Mean
        self.total_mean = copy.deepcopy(self.dataset['train_x']).mean(axis=1).reshape(-1,1)

        # Get within-class mean
        for label in self.train_class_data:
            self.train_class_mean[label] = copy.deepcopy(self.train_class_data[label]).mean(axis=1).reshape(-1,1)

    def split_classes(self):
        """
        Splits and separates the classes

        Parameters
        ----------
        None
        Returns
        -------
        None
        """

        # Get all class labels
        labels = np.unique(self.dataset['train_y'])

        # Seperate Classes
        for label in labels:
            # Get indices for those labels
            indices = []
            for i in range(self.dataset['train_y'].shape[0]):
                indices.append(i) if (self.dataset['train_y'][i] == label) else 0

            # Select class from those indices
            self.train_class_data[label] = copy.deepcopy(self.dataset['train_x'][:,indices])


    def compute_covariance_decomposition(self, matrix):
        """
        Helper function to compute efficient co-varaince eigenvalue decomposition

        Parameters
        ----------
        matrix: matrix
            A numpy matrix

        Returns
        -------
        Eigenvalues, Eigenvectors: matrix
            Eigenvalues and Eigenvectors for the input matrix
        """
        A = copy.deepcopy(matrix)
        cov = (1/matrix.shape[0]) * np.dot(A.T, A)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = copy.deepcopy(np.dot(A,eigenvectors))
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors,axis=0)
        return sort_eigenvalues_eigenvectors( eigenvalues, eigenvectors )

    def eigenvalue_decomposition(self, matrix):
        """
        Helper function to get eigenvalue decomposition of matrix

        Parameters
        ----------
        matrix: matrix
            A numpy matrix

        Returns
        -------
        Eigenvalues, Eigenvectors: matrix
            Eigenvalues and Eigenvectors for the input matrix
        """
        A = copy.deepcopy(matrix)

        # Compute Eigenvalues and Eigenvectors

        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Return eigenvectors and eigenvalues
        return sort_eigenvalues_eigenvectors( eigenvalues, eigenvectors )

    def compute_covariance(self,matrix):
        """
        Helper function to compute the naive covariance for PCA

        Parameters
        ----------
        matrix: matrix
            A numpy matrix

        Returns
        -------
        matrix : matrix
            Dot product of the matrix A*A'
        """
        A = copy.deepcopy(matrix)
        return np.dot(A,A.T)


    def project_matrix(self,matrix,vec):
        """
        Helper function to do a projection into a sub-space

        Parameters
        ----------
        matrix: matrix
            A numpy matrix

        Returns
        -------
        matrix: matrix
            Projected matrix through input vector - vec
        """
        return np.dot(
            copy.deepcopy(vec).T,
            np.dot(
                copy.deepcopy(matrix),
                copy.deepcopy(vec)))

    def transform(self,X):
        """
        Helper function to do a projection into a sub-space using optimal weight
        vectors

        Parameters
        ----------
        X: matrix
            A numpy matrix

        vec: array
            Numpy projection array

        Returns
        -------
        matrix: matrix
            Projected matrix through input vector - vec
        """
        return copy.deepcopy(np.matmul(
                copy.deepcopy(X.T),
                copy.deepcopy(self.w_opt)
            ))

    def fit(self):
        """
        Do the training on the sample data
        1. Computes eigenvectors for the input through PCA
        2. Computes Scatter matrices for LDA
        3. Computes eigenvectors for LDA
        4. Combines theim into optimal wector

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if os.path.exists(os.path.join("w.npy")):
            self.w_opt = np.load("w.npy")
            return

        N = self.dataset['train_x'].shape[1]
        self.M_pca = int(0.5*N)
        self.M_lda = int(0.25*N)

        assert(self.M_pca < self.dataset['train_x'].shape[1]), 'M PCA must be less than N'
        assert(self.M_lda <= self.M_pca), ' M LDA can\'t be greater than M PCA'

        # PCA
        # get eigenvalue decomposition of total covariance matrix (X - Xbar)
        print("Computing PCA eigenvectors...")

        _, pca_eigenvectors = self.compute_covariance_decomposition(copy.deepcopy(self.dataset['train_x']) - self.total_mean)

        # Set M eigenvectors to be w_pca
        if m_pca_type == 0:
            self.w_pca = copy.deepcopy(pca_eigenvectors[:,:min(self.M_pca,pca_eigenvectors.shape[1])])
        if m_pca_type == 1:
            self.w_pca = copy.deepcopy(pca_eigenvectors[:,self.M_pca])

        # Calculate covariance matrices
        class_covariance = []
        class_mean_covariance = []

        print("Computing LDA SW...")

        # Within Class variance
        for key in tqdm(self.train_class_data):
            class_covariance.append( self.compute_covariance( copy.deepcopy(self.train_class_data[key]) - self.train_class_mean[key]) )
        class_covariance = np.array(sum(class_covariance))
        class_covariance = 0.5*(class_covariance + class_covariance.T)
        print(np.linalg.det(class_covariance))
        print("calculating det...")

        # Between class seperation
        print("Computing LDA SB...")

        for key in tqdm(self.train_class_mean):
            class_mean_covariance.append(self.compute_covariance(copy.deepcopy(self.total_mean).reshape(-1,1) - self.train_class_mean[key].reshape(-1,1)) * self.train_class_data[key].shape[1] )
        class_mean_covariance = np.array(sum(class_mean_covariance))
        class_mean_covariance = 0.5*(class_mean_covariance + class_mean_covariance.T)

        # Project covariance matrices to PCA space
        class_covariance        = self.project_matrix(class_covariance, self.w_pca)
        class_mean_covariance   = self.project_matrix(class_mean_covariance, self.w_pca)

        print("Computing LDA eigenvectors...")

        # Calculate optimal projection
        lda_projection = np.dot(np.linalg.pinv(class_covariance),class_mean_covariance)

        # Save LDA projection
        _, lda_eigenvectors = self.eigenvalue_decomposition(lda_projection)

        self.w_lda = copy.deepcopy(lda_eigenvectors[:,:min(self.M_lda,lda_eigenvectors.shape[1])])

        self.w_opt = np.dot(
            copy.deepcopy(self.w_lda.T),
            copy.deepcopy(self.w_pca.T))

        self.w_opt = self.w_opt.T
        np.save("w.npy",self.w_opt)

if __name__ == '__main__':
    # Example Script
    # Load your dataset
    dataset = load_data()

    lda = LDA() # initialise lda class
    lda.dataset['train_x'] = dataset['train_x']
    lda.dataset['train_y'] = dataset['train_y']

    # Setup the class seperation and class means
    lda.run_setup()

    # Select number of vectors for PCA and LDA
    lda.M_pca = 150
    lda.M_lda = 40

    # Fit the subspace
    lda.fit()

    # Get the transformed data
    test_x_transform = lda.transform(dataset['test_x'])
