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

    #err, y_pred = eigenface.run_reconstruction_classifier()

    M = np.arange(0,400,25)

    ########################
    # RECONSTRUCTION ERROR #
    ########################

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
    plt.savefig("results/q1-2/reconstruction_run_time.png", format="png", transparent=True)

    # Error
    plt.plot(M,err)
    plt.ylabel('Error')
    plt.xlabel('Number of Eigenvectors')
    plt.title('Reconstruction Error')
    plt.savefig("results/q1-2/reconstruction_error.png", format="png", transparent=True)


    #############################
    # RECONSTRUCTION COMPARISON #
    #############################

    # Compair Image for different M
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    f.suptitle('Comparison of Reconstructed Faces')
    face = copy.deepcopy(eigenface.train_faces[:,[20]])

    # Original Face
    face_tmp = face + mean
    img = face_tmp.reshape((46,56))
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
    eigenface.select_M_eigenvectors(120, plot=False)
    projected_face = copy.deepcopy(eigenface.project_to_face_space(face))
    reconstructed_face = copy.deepcopy(eigenface.reconstruction(projected_face))

    img = reconstructed_face.reshape((46,56))
    img = np.rot90(img,3)
    ax3.imshow(img, cmap="gray")
    ax3.axis('off')
    ax3.set_title('M=120')

    # Good reconstruction
    eigenface.select_M_eigenvectors(400, plot=False)
    projected_face = copy.deepcopy(eigenface.project_to_face_space(face))
    reconstructed_face = copy.deepcopy(eigenface.reconstruction(projected_face))

    img = reconstructed_face.reshape((46,56))
    img = np.rot90(img,3)
    ax4.imshow(img, cmap="gray")
    ax4.axis('off')
    ax4.set_title('M=400')


    #plt.title('Comparison of reconstruction')
    plt.savefig("results/q1-2/reconstruction_comparison.png", format="png", transparent=True)
    #plt.show()

    ############
    # NN ERROR #
    ############

    err = []
    run_time = []
    for m in M:
        print(m)
        eigenface.M = m
        start = time.time()
        err.append(eigenface.run_nn_classifier()[0])
        end = time.time()
        run_time.append(end-start)

    # Run Time
    plt.plot(M,run_time)
    plt.ylabel('Run Time (s)')
    plt.xlabel('Number of Eigenvectors')
    plt.title('Nearest Neighbour Classifer Run Time')
    plt.savefig("results/q1-2/nn_run_time.png", format="png", transparent=True)


    # Error
    plt.plot(M,err)
    plt.ylabel('Error (MSE)')
    plt.xlabel('Number of Eigenvectors')
    plt.title('Nearest Neighbour Classifer Error')
    plt.savefig("results/q1-2/nn_error.png", format="png", transparent=True)


    #############################################
    # RECONSTRUCTION CLASSIFIER ERROR (FIXED M) #
    #############################################

    err = []
    run_time = []
    M = np.arange(1,8)
    for m in M:
        print(m)
        eigenface.M = m
        start = time.time()
        err.append(eigenface.run_reconstruction_classifier(FIXED_M=True)[0])
        end = time.time()
        run_time.append(end-start)

    # Run Time
    plt.plot(M,run_time)
    plt.ylabel('Run Time (s)')
    plt.xlabel('Number of Eigenvectors')
    plt.title('Reconstruction Classifer Run Time')
    plt.savefig("results/q1-2/reconstruction_classifier_run_time.png", format="png", transparent=True)

    # Error
    plt.plot(M,err)
    plt.ylabel('Error (MSE)')
    plt.xlabel('Number of Eigenvectors')
    plt.title('Reconstruction Classifer Error')
    plt.savefig("results/q1-2/reconstruction_classifier_error.png", format="png", transparent=True)


    ############################################
    # RECONSTRUCTION CLASSIFIER ERROR (CUTOFF) #
    ############################################

    err = []
    run_time = []
    err_cutoff = np.arange(1,70,5)
    for e in err_cutoff:
        start = time.time()
        err.append(eigenface.run_reconstruction_classifier(err_min=e)[0])
        end = time.time()
        run_time.append(end-start)

    # Run Time
    plt.plot(err_cutoff,run_time)
    plt.ylabel('Run Time (s)')
    plt.xlabel('Cutoff for Class-wise Reconstruction Error')
    plt.title('Reconstruction Classifer Run Time')
    plt.savefig("results/q1-2/reconstruction_classifier_run_time.png", format="png", transparent=True)

    # Error
    plt.plot(M,err)
    plt.ylabel('Error (MSE)')
    plt.xlabel('Cutoff for Class-wise Reconstruction Error')
    plt.title('Reconstruction Classifer Error')
    plt.savefig("results/q1-2/reconstruction_classifier_error.png", format="png", transparent=True)

    #########################
    # CLASSIFIER COMPARISON #
    #########################

    # Best, reconstruction classifier
    eigenface.M = 2
    err, y_pred = eigenface.run_reconstruction_classifier()
    # Best, NN classifier
    eigenface.M = 400
    err, y_pred = eigenface.run_nn_classifier()


if __name__ == '__main__':
    main()
