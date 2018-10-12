import numpy as np
from pre_process import load_data
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import matrix_power

class EigenFace:
  def __init__(self):
    # Load pre processed data
    _, Eigenvectors, Eigenvalues, dataset = load_data()

    #Eigenvectors = np.array(Eigenvectors)
    #Eigenvalues = np.array(Eigenvalues)

    print(Eigenvectors[0].shape)
    print(Eigenvalues[0].shape)
    print(dataset[0][0].shape)
    
    # faces from dataset 
    self.test_faces  = dataset[1][0].T
    self.train_faces = dataset[0][0].T
   
    # labels from dataset
    self.test_labels  = dataset[1][1].T
    self.train_labels = dataset[0][1].T
    
    # order and store initial eigenvalues and eigenvectors from training data
    ind = Eigenvalues[0].argsort()
    self.train_eigenvalues  = Eigenvalues[0][ind] 
    self.train_eigenvectors = Eigenvectors[0][ind]

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
    self.select_M_eigenvectors(M, plot=False)
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
  def select_M_eigenvectors(self, M,plot=True):
    if plot:
      plt.plot(eigenvalues)
      plt.show()
    print(self.train_eigenvalues.shape)
    print(self.train_eigenvectors.shape)

    self.m_train_eigenvalues  = self.train_eigenvalues[-M:]
    self.m_train_eigenvectors = self.train_eigenvectors[:,-M:]

  # Do the projection through the eigenvectors
  def project_to_face_space(self, face):
    return np.matmul(face.T, self.m_train_eigenvectors)

  def project_all_to_face_space(self):
    self.train_facespace = [self.project_to_face_space(face) for face in self.train_faces]
    self.projected_test_faces = [self.project_to_face_space(face) for face in self.test_faces]
    return

  def identity_error(labels, labels_correct):
    err = 0
    for i in range(len(labels)):
      if labels[i] != labels_correct[i]:
        err += 1
    #normalise by size of labels
    return err/len(labels)


  def run_nn_classifier(self):
    # empty array to hold label results
    label_results = []

    # select M eigenvectors
    self.select_M_eigenvectors(self.M, plot=False)

    # project to facespace
    self.project_all_to_face_space()
    
    # run nn classifier for every project test face
    for face in tqdm(self.projected_test_faces):
      # get label from nn classifier
      label_results.append(self.nn_classifier(self.test_face, self.projected_facespace,self.train_labels))
    print('error: ',self.identity_error(label_results,self.test_labels))
 
if __name__ == '__main__':
  t = EigenFace()
  t.run_nn_classifier()
