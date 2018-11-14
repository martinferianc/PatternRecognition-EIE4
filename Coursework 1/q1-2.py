import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm
import copy

from pre_process import load_data
from eigenfaces import EigenFace
from profiling import get_process_memory

def main():

    # load data
    mean, eigenvectors, eigenvalues, dataset = load_data()

    # Initialise EigenFace Class
    eigenface = EigenFace(copy.deepcopy(dataset),copy.deepcopy(eigenvectors[0]),mean)

    #err, y_pred = eigenface.run_nn_classifier()

    M = np.arange(0,401,5)
    #M = np.arange(0,200,25)

    ########################
    # RECONSTRUCTION ERROR #
    ########################

    # Obtain reconstruction error as function of M
    err = []
    run_time = []
    mem_consumption = []
    for m in M:
        print(m)
        eigenface.M = m
        start = time.time()
        err.append(eigenface.run_reconstruction())
        end = time.time()
        mem= get_process_memory()
        mem_consumption.append(mem)
        run_time.append(end-start)

    # Run Time
    fig, ax1 = plt.subplots()
    ax1.plot(M,run_time)
    ax1.set_ylabel('Run Time (s)')
    ax1.set_xlabel('Number of Eigenvectors')

    # Memory Consumption
    ax2 = ax1.twinx()
    ax2.plot(M,mem_consumption,'r')
    ax2.set_ylabel('Memory Consumption (%)')

    fig.tight_layout()
    plt.title('Reconstruction Run Time & Memory Consumption')
    plt.savefig("results/q1-2/reconstruction_run_time_and_mem.png", format="png", transparent=True)

    # Error
    plt.figure()
    plt.plot(M,err)
    plt.ylabel('Error')
    plt.xlabel('Number of Eigenvectors')
    plt.title('Reconstruction Error')
    plt.savefig("results/q1-2/reconstruction_error.png", format="png", transparent=True)

    # Run Time
    fig, ax1 = plt.subplots()
    ax1.plot(M,err)
    ax1.set_ylabel('Error')
    ax1.set_xlabel('Number of Eigenvectors')

    # Memory Consumption
    ax2 = ax1.twinx()
    ax2.plot(M,eigenvalues[0][M],'r')
    ax2.set_ylabel('Eigenvalues')

    fig.tight_layout()
    plt.title('Reconstruction Error')
    plt.savefig("results/q1-2/reconstruction_err_eigenvalues.png", format="png", transparent=True)

    #############################
    # RECONSTRUCTION COMPARISON #
    #############################

    # Compair Image for different M
    plt.figure()
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
    mem_consumption = []

    for m in M:
        print(m)
        eigenface.M = m
        start = time.time()
        err.append(eigenface.run_nn_classifier()[0])
        end = time.time()
        mem= get_process_memory()
        mem_consumption.append(mem)
        run_time.append(end-start)

    # Run Time
    fig, ax1 = plt.subplots()
    ax1.plot(M,run_time)
    ax1.set_ylabel('Run Time (s)')
    ax1.set_xlabel('Number of Eigenvectors')

    # Memory Consumption
    ax2 = ax1.twinx()
    ax2.plot(M,mem_consumption,'r')
    ax2.set_ylabel('Memory Consumption (%)')

    fig.tight_layout()
    plt.title('Nearest Neighbour Classifer Run Time & Memory Consumption')
    plt.savefig("results/q1-2/nn_run_time_and_mem.png", format="png", transparent=True)

    # Error
    plt.figure()
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
    plt.figure()
    plt.plot(M,run_time)
    plt.ylabel('Run Time (s)')
    plt.xlabel('Number of Eigenvectors')
    plt.title('Reconstruction Classifer Run Time')
    plt.savefig("results/q1-2/reconstruction_classifier_run_time.png", format="png", transparent=True)

    # Error
    plt.figure()
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
    mem_consumption = []

    err_cutoff = np.arange(1,500,20)
    for e in err_cutoff:
        start = time.time()
        err.append(eigenface.run_reconstruction_classifier(err_min=e)[0])
        end = time.time()
        mem= get_process_memory()
        mem_consumption.append(mem)
        run_time.append(end-start)

    # Run Time
    fig, ax1 = plt.subplots()
    ax1.plot(err_cutoff,run_time)
    ax1.set_ylabel('Run Time (s)')
    ax1.set_xlabel('Number of Eigenvectors')

    # Memory Consumption
    ax2 = ax1.twinx()
    ax2.plot(err_cutoff,mem_consumption,'r')
    ax2.set_ylabel('Memory Consumption (%)')

    fig.tight_layout()
    plt.title('Reconstruction Classifier Run Time & Memory Consumption')
    plt.savefig("results/q1-2/reconstruction_classifier_run_time_and_mem.png", format="png", transparent=True)

    # Error
    plt.figure()
    plt.plot(err_cutoff,err)
    plt.ylabel('Error (MSE)')
    plt.xlabel('Cutoff for Class-wise Reconstruction Error')
    plt.title('Reconstruction Classifer Error')
    plt.savefig("results/q1-2/reconstruction_classifier_error.png", format="png", transparent=True)

    #########################
    # CLASSIFIER COMPARISON #
    #########################

    # Best, reconstruction classifier
    eigenface.M = 2
    err, y_pred = eigenface.run_reconstruction_classifier(err_min=20)

    # Best, NN classifier
    eigenface.M = 400
    err, y_pred = eigenface.run_nn_classifier()

    # find wrong classification
    err_index = 0
    for i in range(3,len(y_pred)):
        if not y_pred[i] == dataset[1][1][i]:
            err_index = i
            break

    correct_face = copy.deepcopy(dataset[1][0][:,[err_index]])
    index = eigenface.nn_classifier_index(eigenface.project_to_face_space(correct_face))
    wrong_face   = copy.deepcopy(dataset[0][0][:,[index]])

    # plot both faces to compare
    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.suptitle('PCA-NN wrong classification comparison')

    img = (correct_face + mean).reshape((46,56))
    img = np.rot90(img,3)
    ax1.imshow(img, cmap="gray")
    ax1.axis('off')
    ax1.set_title('Correct Face')

    img = (wrong_face + mean).reshape((46,56))
    img = np.rot90(img,3)
    ax2.imshow(img, cmap="gray")
    ax2.axis('off')
    ax2.set_title('Wrong Prediction')

    #plt.title('Comparison of reconstruction')
    plt.savefig("results/q1-2/wrong_nn_classifier.png", format="png", transparent=True)


if __name__ == '__main__':
    main()
