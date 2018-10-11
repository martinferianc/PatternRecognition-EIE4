import numpy as np
from pre_process import load_data
import matplotlib.pyplot as plt
from tqdm import tqdm

# Soert and select the M largest eigenvalues and eigenvectors
def select_M_eigenvectors(M, eigenvectors, eigenvalues):
    p = eigenvalues.argsort()
    plt.plot(eigenvalues)
    plt.show()
    eigenvalues = eigenvalues[p]
    eigenvectors = eigenvectors[p]
    return eigenvalues[-M:],eigenvectors[:,-M:]

# Do the projection through the eigenvectors
def project_to_face_space(face, eigenvectors):
    return np.matmul(face.T, eigenvectors)

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
