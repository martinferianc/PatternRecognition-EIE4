import scipy.io
import numpy as np
import copy
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from numpy.linalg import matrix_power
from pre_process import *
from eigenfaces import *

DATA_DIR = "data/"

def nn_classifier(face, training_faces, training_labels):
  nn = training_faces[0]
  label_index = 0
  min_distance =  np.linalg.norm(face - nn)
  for i in range(1,len(training_faces)):
    #get distance between 
    curr_distance = np.linalg.norm(face - training_faces[i])
    if curr_distance < min_distance:
      nn = training_faces[i]
      min_distance = curr_distance
      label_index = i
  return training_labels[label_index]

def identity_error(labels, labels_correct):
  err = 0
  for i in range(len(labels)):
    if labels[i] != labels_correct[i]:
      err += 1
  #normalise by size of labels
  return err/len(labels)

#TODO: define cross validation method to tune M
def tune_M():
  #get training set

  #split into train and validation
  
   

def main():
  # Load all the data
  S, Eigenvectors, Eigenvalues, dataset = load_data()

  # Select the eigenvectors
  M_training_Eigenvalues, M_training_Eigenvectors,  = select_M_eigenvectors(50, Eigenvectors[0], Eigenvalues[0],plot=False)

  test_labels   = dataset[1][1].T
  train_labels  = dataset[0][1].T
 
  test_faces  = dataset[1][0].T
  train_faces = dataset[0][0].T

  label_results = []

  for i in tqdm(range(len(test_faces))):
    #project to face space
    projected_test_face       = project_to_face_space(test_faces[i], M_training_Eigenvectors)
    projected_training_faces  = [project_to_face_space(train_face, M_training_Eigenvectors) for train_face in train_faces]
    # get label from nn classifier
    label_results.append(nn_classifier(projected_test_face, projected_training_faces,train_labels))
  print('error: ',identity_error(label_results,test_labels))

if __name__ == '__main__':
  main()
