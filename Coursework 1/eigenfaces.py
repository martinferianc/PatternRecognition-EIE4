import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.linalg import matrix_power
from pre_process import *
from eigenfaces import *
import copy

class EigenFace:
  def __init__(self,dataset,eigenvectors,mean):

    #store mean
    self.mean = mean

    # faces from dataset
    self.test_faces  = dataset[1][0]
    self.train_faces = dataset[0][0]

    # labels from dataset
    self.test_labels  = dataset[1][1]
    self.train_labels = dataset[0][1]

    # order and store initial eigenvalues and eigenvectors from training data
    self.eigenvectors = eigenvectors

    # temporary variable for number of eigenvectors
    self.M = 50

    # empty set for selected eigenvalues and vectors
    self.m_eigenvectors = []

    # training faces projected by selected eigenvalues
    self.train_facespace = []

    # test faces projected by selected eigenvalues
    self.projected_test_faces = []

  def nn_classifier(self, face):
    nn = copy.deepcopy(self.train_facespace[0])
    label_index = 0
    min_distance =  np.linalg.norm(face - nn)
    for i in range(1,len(self.train_facespace)):
      #get distance between
      curr_distance = np.linalg.norm(face - self.train_facespace[i])
      if curr_distance < min_distance:
        nn = self.train_facespace[i]
        min_distance = curr_distance
        label_index = i
    return self.train_labels[label_index]

  # Sort and select the M largest eigenvalues and eigenvectors
  def select_M_eigenvectors(self, M, plot=False):
    if plot:
      plt.plot(self.eigenvalues)
      plt.show()
    self.m_eigenvectors = copy.deepcopy(self.eigenvectors[:,:M])

  # Do the projection through the eigenvectors
  def project_to_face_space(self, face):
    return np.matmul(face.T, self.m_eigenvectors)

  def project_all_to_face_space(self):
    self.train_facespace = [self.project_to_face_space(copy.deepcopy(face.T)) for face in self.train_faces.T]
    self.projected_test_faces = [self.project_to_face_space(copy.deepcopy(face.T)) for face in self.test_faces.T]
    return

  def plot_img(self,img):
    plt.figure()
    img = img.reshape((46,56))
    img = np.rot90(img,3)
    plt.imshow(img, cmap="gray")
    plt.show()

  def identity_error(self, labels, labels_correct):
    err = 0
    for i in range(len(labels)):
      if labels[i] != labels_correct[i]:
        err += 1
    #normalise by size of labels
    return err/len(labels)

  def mse_error(self, img, img_correct):
    return ((img - img_correct) ** 2).mean(axis=0)

  def reconstruction(self,face):
    projected_face = self.project_to_face_space(face)
    reconstructed_face = copy.deepcopy(self.mean)
    for i in range(len(projected_face)):
      reconstructed_face += projected_face[i] * self.m_eigenvectors[:,[i]]
    return reconstructed_face

  def run_reconstruction(self):
    err_results = []

    # select M eigenvectors
    self.select_M_eigenvectors(self.M, plot=False)

    for face in self.train_faces.T:
      reconstructed_face = self.reconstruction(face.T)
      err_results.append(self.mse_error(face.T,reconstructed_face))
    print('error: ',np.mean(err_results))
    return np.mean(err_results)

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
      label_results.append(self.nn_classifier(face))
    err = self.identity_error(label_results,self.test_labels)
    print('error: ',err)
    return err

if __name__ == '__main__':
  t = EigenFace()
  t.run_nn_classifier()
