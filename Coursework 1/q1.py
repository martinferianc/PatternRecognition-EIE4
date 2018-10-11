import numpy as np
import matplotlib.pyplot as plt
import time

from pre_process import load_mat, remove_mean, separate_data, compute_eigenvalues_eigenvectors

def sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    p = eigenvalues.argsort()
    eigenvalues = eigenvalues[p][::-1]
    eigenvectors = eigenvectors[p][::-1]
    return eigenvalues, eigenvectors

def main():
    # Load the data from the matrix
    X, Y = load_mat("data/face.mat")
    dataset = separate_data((X,Y))

    # Compute and remove mean from the training dataset
    dataset[0][0], mean = remove_mean(dataset[0][0])

    # Remove the same mean from the test dataset
    dataset[1][0] = remove_mean(dataset[1][0], mean)

    # Display the mean face for verification
    plt.figure()
    plt.title('Mean Face')
    mean = mean.reshape((46,56))
    mean = np.rot90(mean,3)
    plt.imshow(mean, cmap="gray")
    plt.savefig("results/q1/mean.png", format="png", transparent=True)

    A = dataset[0][0]
    D, N = A.shape

    start = time.time()
    S_naive = 1/N * np.dot(A, A.T)
    end = time.time()
    print("Calculated S inefficiently, took: {} s".format(end-start))

    start = time.time()
    S_efficient = 1/N * np.dot(A.T, A)
    end = time.time()
    print("Calculated S efficiently, took: {} s".format(end-start))

    start = time.time()
    u_naive,l_naive = compute_eigenvalues_eigenvectors(S_naive)
    end = time.time()
    print("Calculated eigenvectors, eigenvalues inefficiently, took: {} s".format(end-start))

    start = time.time()
    u_efficient,l_efficient = compute_eigenvalues_eigenvectors(S_efficient)
    end = time.time()
    print("Calculated eigenvectors, eigenvalues inefficiently, took: {} s".format(end-start))

    u_naive,l_naive = sort_eigenvalues_eigenvectors(u_naive, l_naive)
    u_efficient, l_efficient = sort_eigenvalues_eigenvectors(u_efficient, l_efficient)
    l_naive = np.real(l_naive)
    l_efficient = np.real(l_efficient)

    plt.figure()
    plt.title('Efficient Eigenvalues')
    plt.xlabel('$l_{m}$ eigenvalue')
    plt.ylabel('Real Value')
    plt.plot(u_efficient)
    plt.savefig('results/q1/eigenvalues_efficient.png',
                format='png', transparent=True)

    plt.figure()
    plt.title('Naive Eigenvalues')
    plt.xlabel('$l_{m}$ eigenvalue')
    plt.ylabel('Real Value')
    plt.plot(u_naive)
    plt.savefig('results/q1/eigenvalues_naive.png',
                format='png', transparent=True)

    plt.figure()
    plt.title('Combined Efficient and Naive Eigenvalues')
    plt.xlabel('$l_{m}$ eigenvalue')
    plt.ylabel('Real Value')
    plt.plot(u_naive, label="Naive implementation")
    plt.plot(u_efficient, label="Efficient implementation")
    plt.legend()
    plt.savefig('results/q1/eigenvalues_efficient_naive.png',
                format='png', transparent=True)

    plt.close()


if __name__ == '__main__':
    main()
