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
    eigenface = EigenFace(copy.deepcopy(dataset),copy.deepcopy(eigenvectors[0]),mean)
    eigenface.M = 200
    #eigenface.run_reconstruction_classifier()
    #eigenface.run_nn_classifier()

    M = np.arange(0,400,25)

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

    # Run Time
    plt.plot(M,run_time)
    plt.ylabel('Run Time (s)')
    plt.xlabel('Number of Eigenvectors')
    plt.title('Reconstruction Run Time')
    plt.show()

    # Error
    plt.plot(M,err)
    plt.ylabel('Error')
    plt.xlabel('Number of Eigenvectors')
    plt.title('Reconstruction Error')
    plt.show()

    '''

    #############################
    # RECONSTRUCTION COMPARISON #
    #############################

    # Compair Image for different M
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    f.suptitle('Comparison of Reconstructed Faces')
    face = copy.deepcopy(eigenface.train_faces[:,[20]])

    # Original Face
    face_tmp = face + mean
    img = face_tmp.reshape((46,56))
    print(img)
    img = np.rot90(img,3)
    ax1.imshow(img, cmap="gray")
    ax1.axis('off')
    ax1.set_title('Original')


    # Bad reconstruction
    eigenface.select_M_eigenvectors(5, plot=False)
    projected_face = copy.deepcopy(eigenface.project_to_face_space(face))
    reconstructed_face = copy.deepcopy(eigenface.reconstruction(projected_face))

    img = reconstructed_face.reshape((46,56))
    img = np.rot90(img,3)
    ax2.imshow(img, cmap="gray")
    ax2.axis('off')
    ax2.set_title('M=5')


    # Good reconstruction
    eigenface.select_M_eigenvectors(400, plot=False)
    projected_face = copy.deepcopy(eigenface.project_to_face_space(face))
    reconstructed_face = copy.deepcopy(eigenface.reconstruction(projected_face))

    img = reconstructed_face.reshape((46,56))
    print(img)
    img = np.rot90(img,3)
    ax3.imshow(img, cmap="gray")
    ax3.axis('off')
    ax3.set_title('M=400')


    #plt.title('Comparison of reconstruction')
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
