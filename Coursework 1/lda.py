import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import matrix_power
from pre_process import *
from eigenfaces import *
import copy
from pre_process import sort_eigenvalues_eigenvectors
from sklearn.model_selection import train_test_split

from pre_process_raw import load_data

class LDA:
    def __init__(self,dataset_filename='data/face.mat',LOAD=False):

        # Dataset
        # - Stores the raw dataset
        # - note, don't remove mean or anything from this data
        self.dataset = {
            'train_x': [],
            'train_y': [],
            'test_x': [],
            'test_y': []
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
        self.M_pca = 50
        self.M_lda = 50

        # Load data
        # note - gets dataset for faces (CW 1)
        if LOAD:
            self.get_dataset(dataset_filename)

    # NOT NEEDED
    def get_normal_dataset(self,data,labels):
        for i in range(data.shape[1]):
            label = labels[i][0]
            tmp = data[:,i] - self.train_class_mean[ label ].ravel()
            data[:,i] = tmp

        return data

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
        #labels = set(self.dataset['train_y'].tolist())
        labels = np.unique(self.dataset['train_y'])

        # Seperate Classes
        for label in labels:
            # Get indices for those labels
            indices = []
            for i in range(self.dataset['train_y'].shape[0]):
                indices.append(i) if (self.dataset['train_y'][i] == label) else 0

            # Select class from those indices
            self.train_class_data[label] = copy.deepcopy(self.dataset['train_x'][:,indices])

    # simple function to get size of each class in training set
    def get_class_sizes(self):

        class_sizes = {}

        for label in self.train_class_data:
            class_sizes[label] = self.train_class_data[label].shape[1]

        return class_sizes

    # gets a dataset from a .mat file and does 80-20 split
    def get_dataset(self,filename):
        # Load data from file
        X, [y] = load_mat(filename)

        # Perform train,test split
        self.dataset['train_x'], self.dataset['test_x'], self.dataset['train_y'], self.dataset['test_y'] = train_test_split(X.T, y, test_size=0.2,stratify=y)

        # Adjust data orientation
        self.dataset['train_x'] = self.dataset['train_x'].T
        self.dataset['test_x']  = self.dataset['test_x'].T

        # Seperate Classes
        ## Get set of all labels
        labels = set(self.dataset['train_y'])

        for label in labels:
            # Get indices for those labels
            indices = []
            for i in range(self.dataset['train_y'].shape[0]):
                indices.append(i) if (self.dataset['train_y'][i] == label) else 0

            # Select class from those indices
            self.train_class_data[label] = copy.deepcopy(self.dataset['train_x'][:,indices])

        # Get mean for each class
        for label in self.train_class_data:
            self.train_class_mean[label] = copy.deepcopy(self.train_class_data[label]).mean(axis=1).reshape(-1,1)

        # Get total mean
        self.total_mean = copy.deepcopy(self.dataset['train_x']).mean(axis=1).reshape(-1,1)

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

    # get's PCA (DEPRECIATED)
    def get_pca(self, M_pca):
        # get eigenvalue decomposition of total covariance matrix (X - Xbar)
        eigenvalues, eigenvectors = self.compute_covariance_decomposition(copy.deepcopy(self.dataset['train_x']) - self.total_mean)
        # set M eigenvectors to be w_pca
        self.w_pca = copy.deepcopy(eigenvectors[:,:min(M_pca,eigenvectors.shape[1])])
        return self.w_pca


    def project_matrix(self,matrix,vec):
        return np.dot(
            copy.deepcopy(vec).T,
            np.dot(
                copy.deepcopy(matrix),
                copy.deepcopy(vec)))

    # IMPORTANT : function to do PCA-LDA transformation
    def transform(self,X):
        return copy.deepcopy(np.dot(
                copy.deepcopy(X.T),
                copy.deepcopy(self.w_opt)
            ))

    # IMPORTANT : Equivalent of 'fit' function
    def run_pca_lda(self,m_pca_type=0,m_lda_type=0): # 0 - single value, 1 - list

        #if m_pca_type == 0:
        #    assert(self.M_pca < self.dataset['train_x'].shape[1]), 'M PCA must be less than N'
        #if m_lda_type == 0:
        #    assert(self.M_lda <= self.M_pca), ' M LDA can\'t be greater than M PCA'

        # PCA
        # get eigenvalue decomposition of total covariance matrix (X - Xbar)
        _, pca_eigenvectors = self.compute_covariance_decomposition(copy.deepcopy(self.dataset['train_x']) - self.total_mean)

        # set M eigenvectors to be w_pca
        if m_pca_type == 0:
            self.w_pca = copy.deepcopy(pca_eigenvectors[:,:min(self.M_pca,pca_eigenvectors.shape[1])])
        if m_pca_type == 1:
            self.w_pca = copy.deepcopy(pca_eigenvectors[:,self.M_pca])

        # Calculate covariance matrices
        class_covariance = []
        class_mean_covariance = []

        # Within Class variance
        for key in tqdm(self.train_class_data):
            class_covariance.append( self.compute_covariance( copy.deepcopy(self.train_class_data[key]) - self.train_class_mean[key]) )
        class_covariance = sum(class_covariance)

        # Between class seperation
        for key in tqdm(self.train_class_mean):
            class_mean_covariance.append(self.compute_covariance(copy.deepcopy(self.total_mean).reshape(-1,1) - self.train_class_mean[key].reshape(-1,1)) * self.train_class_data[key].shape[1] )
        class_mean_covariance = sum(class_mean_covariance)

        # get rank of matrices
        #print('Before PCA projection ... \n')
        #print('class covariance rank = ',np.linalg.matrix_rank(class_covariance))
        #print('class mean covariance rank = ',np.linalg.matrix_rank(class_mean_covariance))

        # Project covariance matrices to PCA space
        class_covariance        = self.project_matrix(class_covariance, self.w_pca)
        class_mean_covariance   = self.project_matrix(class_mean_covariance, self.w_pca)
        #class_covariance        = (self.w_pca.T).dot(class_covariance)
        #class_mean_covariance   = (self.w_pca.T).dot(class_mean_covariance)

        #print('\n\nAfter PCA projection ... \n')
        #print('class covariance rank = ',np.linalg.matrix_rank(class_covariance))
        #print('class mean covariance rank = ',np.linalg.matrix_rank(class_mean_covariance))

        # Calculate optimal projection
        lda_projection = np.dot(np.linalg.inv(class_covariance),class_mean_covariance)

        # Save LDA projection
        _, lda_eigenvectors = self.eigenvalue_decomposition(lda_projection)
        #self.w_lda  = self.w_lda[:,:M_lda]
        #self.w_lda  = self.w_lda / np.linalg.norm(self.w_lda, axis=0)
        if m_lda_type == 0:
            self.w_lda = copy.deepcopy(lda_eigenvectors[:,:min(self.M_lda,lda_eigenvectors.shape[1])])
        if m_lda_type == 1:
            self.w_lda = copy.deepcopy(lda_eigenvectors[:,self.M_lda])

        self.w_opt = np.dot(
            copy.deepcopy(self.w_lda.T),
            copy.deepcopy(self.w_pca.T))

        self.w_opt = self.w_opt.T

    def project(self,img,vec):
        return copy.deepcopy(np.dot(copy.deepcopy(img.T), copy.deepcopy(vec)))

    def nn_classifier(self, face, facespace, labels):
        nn = copy.deepcopy(facespace[0])
        label_index = 0
        min_distance =  np.linalg.norm(face - nn)
        for i in range(1,facespace.shape[0]):
            #get distance between
            curr_distance = np.linalg.norm(face - facespace[i])
            if curr_distance < min_distance:
                nn = facespace[i]
                min_distance = curr_distance
                label_index = i
        return labels[label_index]

    def run_nn_classifier(self):
        # empty array to hold label results
        label_results = []

        # project faces
        projected_test_x  = self.project((copy.deepcopy(self.dataset['test_x'])),copy.deepcopy(self.w_opt))
        projected_train_x = self.project((copy.deepcopy(self.dataset['train_x'])),copy.deepcopy(self.w_opt))
        # run nn classifier for every project test face
        for face in tqdm(projected_test_x):
            # get label from nn classifier
            label_results.append(self.nn_classifier(face,projected_train_x,self.dataset['train_y']))
        err = self.identity_error(label_results,self.dataset['test_y'])
        print('error: ',err)
        return err , label_results

    def identity_error(self, labels, labels_correct):
        err = 0
        for i in range(len(labels)):
            if labels[i] != labels_correct[i]:
                err += 1
        #normalise by size of labels
        return err/len(labels)

if __name__ == '__main__':

    # EXAMPLE SCRIPT

    # load your dataset
    dataset = load_data() # contains train_x, train_y, test_x, test_y

    lda = LDA() # initialise lda class
    lda.dataset['train_x'] = dataset['train_x']
    lda.dataset['train_y'] = dataset['train_y']
    lda.dataset['test_x']  = dataset['test_x']
    lda.dataset['test_y']  = dataset['test_y']

    # setup the class seperation and class means
    lda.run_setup()

    # Select number of vectors for PCA and LDA
    lda.M_pca = 150 # TODO: change for dataset
    lda.M_lda = 40  # TODO: change for datset

    # fit the subspace
    lda.run_pca_lda()

    # get the transformed data
    test_x_transform = lda.transform(dataset['test_x'])
