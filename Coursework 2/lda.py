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


    # splits dataset between classes and gets mean
    def run_setup(self):
        self.split_classes()
        self.get_mean()

    # gets both class mean and total mean
    def get_mean(self):
        # get Total Mean
        self.total_mean = copy.deepcopy(self.dataset['train_x']).mean(axis=1).reshape(-1,1)

        # get within-class mean
        for label in self.train_class_data:
            self.train_class_mean[label] = copy.deepcopy(self.train_class_data[label]).mean(axis=1).reshape(-1,1)

    # seperates classes
    def split_classes(self):
        # Get all class labels
        labels = np.unique(self.dataset['train_y'])

        # Seperate Classes
        for label in labels:
            # Get indices for those labels
            indices = []
            for i in range(self.dataset['train_y'].shape[0]):
                indices.append(i) if (self.dataset['train_y'][i] == label) else 0

            # Select class from those indices
            #print(indices)
            self.train_class_data[label] = copy.deepcopy(self.dataset['train_x'][:,indices])

    # simple function to get size of each class in training set
    def get_class_sizes(self):
        class_sizes = {}

        for label in self.train_class_data:
            class_sizes[label] = self.train_class_data[label].shape[1]

        return class_sizes

    # helper function to compute efficient co-varaince eigenvalue decomposition
    def compute_covariance_decomposition(self, matrix):
        A = copy.deepcopy(matrix)
        cov = (1/matrix.shape[0]) * np.dot(A.T, A)
        #print('Before PCA projection ... \n')
        #print('class covariance rank = ',np.linalg.matrix_rank(cov))
        # Compute Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # Transform Eigenvectors to the Face plane
        eigenvectors = copy.deepcopy(np.dot(A,eigenvectors))
        # Normalise Eigenvectors
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors,axis=0)
        # return eigenvectors and eigenvalues
        return sort_eigenvalues_eigenvectors( eigenvalues, eigenvectors )

    # helper function to get eigenvalue decomposition of matrix
    def eigenvalue_decomposition(self, matrix):
        A = copy.deepcopy(matrix)
        # Compute Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # return eigenvectors and eigenvalues
        return sort_eigenvalues_eigenvectors( eigenvalues, eigenvectors )

    # helper function to compute co-variance
    def compute_covariance(self,matrix):
        A = copy.deepcopy(matrix)
        return np.dot(A,A.T)


    def project_matrix(self,matrix,vec):
        return np.dot(
            copy.deepcopy(vec).T,
            np.dot(
                copy.deepcopy(matrix),
                copy.deepcopy(vec)))

    # IMPORTANT : function to do PCA-LDA transformation
    def transform(self,X):
        return copy.deepcopy(np.matmul(
                copy.deepcopy(X.T),
                copy.deepcopy(self.w_opt)
            ))

    # IMPORTANT : Equivalent of 'fit' function
    def run_pca_lda(self,m_pca_type=0,m_lda_type=0): # 0 - single value, 1 - list
        if os.path.exists(os.path.join("w.npy")):
            self.w_opt = np.load("w.npy")
            return

        N = self.dataset['train_x'].shape[1]
        self.M_pca = int(0.5*N)
        self.M_lda = int(0.25*N)

        if m_pca_type == 0:
            assert(self.M_pca < self.dataset['train_x'].shape[1]), 'M PCA must be less than N'
        if m_lda_type == 0:
            assert(self.M_lda <= self.M_pca), ' M LDA can\'t be greater than M PCA'

        # PCA
        # get eigenvalue decomposition of total covariance matrix (X - Xbar)
        print("Computing PCA eigenvectors...")

        _, pca_eigenvectors = self.compute_covariance_decomposition(copy.deepcopy(self.dataset['train_x']) - self.total_mean)

        # set M eigenvectors to be w_pca
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
        print(np.linalg.det(class_mean_covariance))
        # Project covariance matrices to PCA space
        class_covariance        = self.project_matrix(class_covariance, self.w_pca)
        class_mean_covariance   = self.project_matrix(class_mean_covariance, self.w_pca)

        print("Computing LDA eigenvectors...")
        # Calculate optimal projection
        lda_projection = np.dot(np.linalg.pinv(class_covariance),class_mean_covariance)

        # Save LDA projection
        _, lda_eigenvectors = self.eigenvalue_decomposition(lda_projection)

        if m_lda_type == 0:
            self.w_lda = copy.deepcopy(lda_eigenvectors[:,:min(self.M_lda,lda_eigenvectors.shape[1])])
        if m_lda_type == 1:
            self.w_lda = copy.deepcopy(lda_eigenvectors[:,self.M_lda])

        self.w_opt = np.dot(
            copy.deepcopy(self.w_lda.T),
            copy.deepcopy(self.w_pca.T))

        self.w_opt = self.w_opt.T
        np.save("w.npy",self.w_opt)

    def project(self,img,vec):
        return copy.deepcopy(np.dot(copy.deepcopy(img.T), copy.deepcopy(vec)))

if __name__ == '__main__':

    # EXAMPLE SCRIPT

    # load your dataset
    dataset = load_data() # contains train_x, train_y, test_x, test_y

    lda = LDA() # initialise lda class
    lda.dataset['train_x'] = dataset['train_x']
    lda.dataset['train_y'] = dataset['train_y']

    # setup the class seperation and class means
    lda.run_setup()

    # Select number of vectors for PCA and LDA
    lda.M_pca = 150 # TODO: change for dataset
    lda.M_lda = 40  # TODO: change for datset

    # fit the subspace
    lda.run_pca_lda()

    # get the transformed data
    test_x_transform = lda.transform(dataset['test_x'])
