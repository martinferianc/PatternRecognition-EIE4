import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import matrix_power
from pre_process import *
from eigenfaces import *
import copy
from pre_process import sort_eigenvalues_eigenvectors
from sklearn.model_selection import train_test_split

class LDA:
    def __init__(self,dataset_filename):

        # Dataset
        self.dataset = {'train_x': [], 'train_y': [], 'test_x': [], 'test_y': []}

        # Training Data class seperation
        self.train_class_data = {}
        self.train_class_mean = {}

        # Total mean
        self.total_mean = []

        # Projection Matrices
        self.w_pca = []
        self.w_lda = []
        self.w_opt = []

        # Load data
        self.get_dataset(dataset_filename)

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

    def compute_covariance_decomposition(self, matrix):
        A = copy.deepcopy(matrix)
        cov = np.dot(A.T, A)
        # Compute Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # Transform Eigenvectors to the Face plane
        eigenvectors = copy.deepcopy(np.matmul(A,eigenvectors))
        # Normalise Eigenvectors
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors,axis=0)
        # return eigenvectors and eigenvalues
        return sort_eigenvalues_eigenvectors( eigenvalues, eigenvectors )

    def compute_covariance(self,matrix):
        A = copy.deepcopy(matrix)
        return np.dot(A,A.T)

    def get_pca(self, M_pca):
        # get eigenvalue decomposition of total covariance matrix (X - Xbar)
        eigenvalues, eigenvectors = self.compute_covariance_decomposition(self.dataset['train_x'] - self.total_mean)
        # set M eigenvectors to be w_pca
        self.w_pca = eigenvectors[:,:M_pca]
        return self.w_pca

    def run_pca_lda(self):

        # Get PCA transform
        self.get_pca(200)

        # Calculate covariance matrices
        class_covariance = []
        class_mean_covariance = []

        for key in tqdm(self.train_class_data):
            class_covariance.append(self.compute_covariance(self.train_class_data[key] - self.train_class_mean[key]))
        class_covariance = sum(class_covariance)

        for key in tqdm(self.train_class_mean):
            class_mean_covariance.append(self.compute_covariance(self.total_mean - self.train_class_mean[key]))
        class_mean_covariance = sum(class_mean_covariance)

        # Project covariance matrices to PCA space
        class_covariance        = np.matmul(self.w_pca.T, np.matmul( class_covariance, self.w_pca))
        class_mean_covariance   = np.matmul(self.w_pca.T, np.matmul(class_mean_covariance, self.w_pca))

        # Calculate optimal projection
        lda_projection = np.matmul(np.linalg.pinv(class_covariance),class_mean_covariance)

        # Save LDA projection
        self.w_lda = lda_projection
        self.w_lda = self.w_lda[:,:100]
        self.w_lda = self.w_lda / np.linalg.norm(self.w_lda, axis=0)

        self.w_opt = np.matmul(self.w_lda.T,self.w_pca.T)
        self.w_opt = self.w_opt.T

        self.w_opt = self.w_opt / np.linalg.norm(self.w_opt,axis=0)

        #self.w_opt = self.w_opt[:,:100]

    def project(self,img,vec):
        return copy.deepcopy(np.dot(img.T, vec))

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
        projected_test_x  = self.project((self.dataset['test_x'] -self.total_mean),self.w_opt)
        projected_train_x = self.project((self.dataset['train_x']-self.total_mean),self.w_opt)

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
    lda = LDA('data/face.mat')
    lda.run_pca_lda()
    lda.run_nn_classifier()
