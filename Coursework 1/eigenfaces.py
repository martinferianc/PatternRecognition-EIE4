import numpy as np
from pre_process import load_data
import matplotlib.pyplot as plt
from tqdm import tqdm

def best_eigenvectors_cutoff(eigenvalues,eigenvectors,cutoff):
    #eigenvalues ordered
    M = len(eigenvalues) 
    eigenvalues_pwr = np.square(np.absolute(eigenvalues)) 
    for i in range(len(eigenvalues_pwr)):
        #find the eigenvalue that's below the cutoff
        if eigenvalues_pwr[i] < cutoff:
            M=i
            break
    return M

def best_eigenvectors_gradient(eigenvalues,eigenvectors,gradient):
    #eigenvalues ordered
    M = len(eigenvalues) 
    eigenvalues_pwr = np.square(np.absolute(eigenvalues)) 
    for i in range(1,len(eigenvalues_pwr)):
        #find the gradient below the hyperparameter
        if abs(eigenvalues_pwr[i] - eigenvalues_pwr[i-1]) < gradient:
            M=i
            break
    return M 


# Soert and select the M largest eigenvalues and eigenvectors
def select_M_eigenvectors(M, eigenvectors, eigenvalues,plot=True):
    p = eigenvalues.argsort()
    if plot:
        plt.plot(eigenvalues)
        plt.show()
    eigenvalues = eigenvalues[p]
    eigenvectors = eigenvectors[p]
    return eigenvalues[-M:],eigenvectors[:,-M:]

# Do the projection through the eigenvectors
def project_to_face_space(face, eigenvectors):
    return np.matmul(eigenvectors.transpose(), face)

def main():
    # Load all the data
    S, Eigenvectors, Eigenvalues, dataset = load_data()

    # Select the eigenvectors
    M_training_Eigenvalues, M_training_Eigenvectors,  = select_M_eigenvectors(5, Eigenvectors[0], Eigenvalues[0],plot=False)

    # # TODO: This can be optimized with a for loop to find the best M in terms of time
    # memory and accuracy
    # Initialize the error to 0
    error = 0
    test_results = []

    test_labels = dataset[1][1].T
    train_labels = dataset[0][1].T

    # Iterate over all the test images
    index_test = 0

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
        for training_face in dataset[0][0].T:
            # Get the projection of the training image into the face space
            projected_training_face = project_to_face_space(training_face, M_training_Eigenvectors)

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
