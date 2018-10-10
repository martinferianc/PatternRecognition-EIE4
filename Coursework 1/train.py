import scipy.io
import numpy as np
import copy
import os
from sklearn.model_selection import train_test_split
from numpy.linalg import matrix_power
import pre_process

DATA_DIR = "data/"


# function to find best M eigenvalues (step 7)
# input           : eigenvectors, eigenvalues (N)
# output          : eigenvectors, eigenvalues (M)
# hyperparameter  : cutoff
def best_eigenvectors_cutoff(eigenvalues,eigenvectors,cutoff):
  #eigenvalues ordered
  M = len(eigenvalues) 
  eigenvalues_pwr = np.square(np.absolute(eigenvalues)) 
  for i in range(len(eigenvalues_pwr)):
    #find the eigenvalue that's below the cutoff
    if eigenvalues_pwr[i] < cutoff:
      M=i
      break
  return [eigenvalues[0:M], eigenvectors[0:M]]

def best_eigenvectors_gradient(eigenvalues,eigenvectors,gradient):
  #eigenvalues ordered
  M = len(eigenvalues) 
  eigenvalues_pwr = np.square(np.absolute(eigenvalues)) 
  for i in range(1,len(eigenvalues_pwr)):
    #find the gradient below the hyperparameter
    if abs(eigenvalues_pwr[i] - eigenvalues_pwr[i-1]) < gradient:
      M=i
      break
  return [eigenvalues[0:M], eigenvectors[0:M]]

def main():
  eigenvectors = np.load(DATA_DIR +"processed/eigenvectors/training.npy")
  eigenvalues  = np.load(DATA_DIR +"processed/eigenvalues/training.npy")
    
  #[best_eigenvalues,best_eigenvectors] = best_eigenvectors_cutoff(eigenvalues,eigenvectors,10)
  [best_eigenvalues,best_eigenvectors] = best_eigenvectors_gradient(eigenvalues,eigenvectors,1000)

  print(best_eigenvalues)
  print(best_eigenvectors)


if __name__ == '__main__':
  main()
