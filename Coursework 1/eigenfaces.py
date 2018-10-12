import numpy as np
from pre_process import load_data
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import matrix_power

class EigenFace:
  def __init__(self):
    # Load pre processed data
    _, Eigenvectors, Eigenvalues, dataset = load_data()
    
    # faces from dataset 
    self.test_faces  = dataset[1][0].T
    self.train_faces = dataset[0][0].T
   
    # labels from dataset
    self.test_labels  = dataset[1][1].T
    self.train_labels = dataset[0][1].T
    
    # order and store initial eigenvalues and eigenvectors from training data
    ind = Eigenvalues.argsort()
    self.train_eigenvalues  = Eigenvalues[ind] 
    self.train_eigenvectors = Eigenvectors[ind]

    # temporary variable for number of eigenvectors
    self.M = 200

    # empty set for selected eigenvalues and vectors
    self.m_train_eigenvalues  = []
    self.m_train_eigenvectors = []

    # training faces projected by selected eigenvalues
    self.train_facespace = []
    
    # test faces projected by selected eigenvalues
    self.projected_test_faces = []

  # Function to get the largest eigenvectors for a given eigenvalue cutoff
  # input : eigenvalue cutoff
  # return: number of eigenvectors
  def best_eigenvectors_cutoff(self,cutoff):
    #eigenvalues ordered
    M = 1 
    eigenvalues_pwr = np.square(np.absolute(self.train_eigenvalues)) 
    for i in range(len(eigenvalues_pwr)):
      #find the eigenvalue that's below the cutoff
      if eigenvalues_pwr[i] < cutoff:
        M=i
        break
    # set M and m_train_eigenvalues
    self.M = M
    self.select_M_eigenvectors(M, plot=False):
    return M

  def best_eigenvectors_gradient(self,eigenvalues,gradient):
    #eigenvalues ordered
    M = len(eigenvalues) 
    eigenvalues_pwr = np.square(np.absolute(eigenvalues)) 
    for i in range(1,len(eigenvalues_pwr)):
      #find the gradient below the hyperparameter
      if abs(eigenvalues_pwr[i] - eigenvalues_pwr[i-1]) < gradient:
        M=i
        break
    return M 

  def nn_classifier(self, face, train_facespace, train_labels):
    nn = train_facespace[0]
    label_index = 0
    min_distance =  np.linalg.norm(face - nn)
    for i in range(1,len(train_facespace)):
      #get distance between 
      curr_distance = np.linalg.norm(face - train_facespace[i])
      if curr_distance < min_distance:
        nn = train_facespace[i]
        min_distance = curr_distance
        label_index = i
    return training_labels[label_index]

  # Soert and select the M largest eigenvalues and eigenvectors
  def select_M_eigenvectors(self, M, eigenvectors, eigenvalues,plot=True):
    if plot:
      plt.plot(eigenvalues)
      plt.show()
    self.m_train_eigenvalues  = self.train_eigenvalues[-M:]
    self.m_train_eigenvectors = self.train_eigenvectors[:,-M:]

  # Do the projection through the eigenvectors
  def project_to_face_space(self, face):
    return np.matmul(face.T, self.m_train_eigenvectors)

  def run_nn_classifier(self):
    pass

def main():
    # Load all the data
    S, Eigenvectors, Eigenvalues, dataset = load_data()

    # Select the eigenvectors
    M_training_Eigenvalues, M_training_Eigenvectors,  = select_M_eigenvectors(200, Eigenvectors[0], Eigenvalues[0])

    # # TODO: This can be optimized with a for loop to find the best M in terms of time
    # memory and accuracy
    # Initialize the error to 0
    error = 0
    test_results = []

    test_labels = dataset[1][1].T
    train_labels = dataset[0][1].T

    # Iterate over all the test images
    index_test = 0

    # Project all the training faces into the face space and cache them
    projected_training_faces = []
    for training_face in dataset[0][0].T:
        # Get the projection of the training image into the face space
        projected_training_faces.append(project_to_face_space(training_face, M_training_Eigenvectors))

    # Do this for every test face
    for test_face in tqdm(dataset[1][0].T):

        # Get the projection into the face space
        projected_test_face = project_to_face_space(test_face, M_training_Eigenvectors)

        # Initialize the euclidian distance as infinity
        max_distance = float("inf")

        # Remember the indices for the training images
        # and for the best neighbout
        index_train = 0
        index = 0
        for projected_training_face in projected_training_faces:

            # Calculate the Euclidian distance
            distance = np.linalg.norm(projected_test_face - projected_training_face)

            # If the distance is smaller remember it
            if distance < max_distance:
                max_distance = distance
                index = index_train
            index_train+=1

        test_results.append(train_labels[index])
        if test_labels[index_test]!=train_labels[index]:
            error+=1
        index_test+=1
    print("Error {}".format(error/len(test_labels)))


if __name__ == '__main__':
    main()
