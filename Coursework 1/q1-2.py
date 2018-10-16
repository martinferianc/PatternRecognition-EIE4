import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm
import copy

from pre_process import load_data
from eigenfaces import EigenFace

def main():

    # load data
    mean, eigenvectors, eigenvalues, dataset = load_data()

    # Initialise EigenFace Class
    eigenface = EigenFace(dataset,eigenvectors[0],mean)

    M = np.arange(50,400,10)

    ########################
    # RECONSTRUCTION ERROR #
    ########################

    '''

    # Obtain reconstruction error as function of M
    err = []
    run_time = []
    for m in M:
        print(m)
        eigenface.M = m
        start = time.time()
        err.append(eigenface.run_reconstruction())
        end = time.time()
        run_time.append(end-start)
    plt.plot(M,run_time)
    plt.show()
    plt.plot(M,err)
    plt.show()

    '''

    #############################
    # RECONSTRUCTION COMPARISON #
    #############################

    # Compair Image for different M
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    face = copy.deepcopy(eigenface.test_faces[:,[1]])

    img = face.reshape((46,56))
    img = np.rot90(img,3)
    ax1.imshow(img, cmap="gray")
    ax1.axis('off')

    eigenface.select_M_eigenvectors(5, plot=False)
    projected_face = copy.deepcopy(eigenface.project_to_face_space(face))
    reconstructed_face = copy.deepcopy(eigenface.reconstruction(projected_face))

    img = reconstructed_face.reshape((46,56))
    img = np.rot90(img,3)
    ax2.imshow(img, cmap="gray")
    ax2.axis('off')

    eigenface.select_M_eigenvectors(400, plot=False)
    projected_face = copy.deepcopy(eigenface.project_to_face_space(face))
    reconstructed_face = copy.deepcopy(eigenface.reconstruction(projected_face))

    img = reconstructed_face.reshape((46,56))
    img = np.rot90(img,3)
    ax3.imshow(img, cmap="gray")
    ax3.axis('off')

    plt.show()

    ############
    # NN ERROR #
    ############

    err = []
    run_time = []
    for m in M:
        print(m)
        eigenface.M = m
        start = time.time()
        err.append(eigenface.run_nn_classifier())
        end = time.time()
        run_time.append(end-start)

if __name__ == '__main__':
    main()
