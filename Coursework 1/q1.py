import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm

from pre_process import load_mat, remove_mean, separate_data, compute_eigenvalues_eigenvectors

def sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    p = eigenvalues.argsort()
    eigenvalues = eigenvalues[p][::-1]
    eigenvectors = eigenvectors[p][::-1]
    return eigenvalues, eigenvectors

def non_zero(eigenvalues,lim=0.001):
    count = 0
    print(eigenvalues.shape)
    for i in eigenvalues:
        if i<lim:
            count+=1
    return count

def compare(a1, a2, lim=0.001):
    N = len(a1) if len(a1) < len(a2) else a2
    for i in range(N):
        if np.abs(a1[i]-a2[i])>lim:
            return False
    return True

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
    print(A.shape)
    start = time.time()
    l_naive,u_naive = compute_eigenvalues_eigenvectors(S_naive)
    end = time.time()
    print("Calculated eigenvectors, eigenvalues efficiently, took: {} s".format(end-start))

    start = time.time()
    l_efficient,u_efficient = compute_eigenvalues_eigenvectors(S_efficient)
    end = time.time()
    print("Calculated eigenvectors, eigenvalues inefficiently, took: {} s".format(end-start))

    l_naive,u_naive = sort_eigenvalues_eigenvectors(l_naive, u_naive)
    l_efficient, u_efficient = sort_eigenvalues_eigenvectors(l_efficient, u_efficient)
    l_naive = np.real(l_naive)
    l_efficient = np.real(l_efficient)
    print("Non-zero eigenvalues inefficiently {}".format(non_zero(l_naive)))
    print("Non-zero eigenvalues efficiently {}".format(non_zero(l_efficient)))

    print("Are eigenvalues for both the same? {}".format(compare(l_efficient, l_naive)))

    l_naive_sum = np.sum(l_naive)
    k = [0] * N

    for i in range(N):
        for j in range(i):
            k[i]+=l_naive[j]
        k[i]/=l_naive_sum
    k = np.array(k)
    print("Relative accurracy for M=10: {}, M=50: {}, M = 100: {}, M=200: {}".format(k[10], k[50], k[100], k[200]))

    plt.figure()
    plt.title('Relative Reconstruction Error')
    plt.xlabel('$l_{m}$ eigenvalue sum')
    plt.ylabel('Accuracy')
    plt.plot(k)
    plt.savefig('results/q1/relative_accuracy.png',
                format='png', transparent=True)

    plt.figure()
    plt.title('Efficient Eigenvalues')
    plt.xlabel('$l_{m}$ eigenvalue')
    plt.ylabel('Real Value')
    plt.plot(l_efficient)
    plt.savefig('results/q1/eigenvalues_efficient.png',
                format='png', transparent=True)

    plt.figure()
    plt.title('Naive Eigenvalues')
    plt.xlabel('$l_{m}$ eigenvalue')
    plt.ylabel('Real Value')
    plt.plot(l_naive)
    plt.savefig('results/q1/eigenvalues_naive.png',
                format='png', transparent=True)

    plt.figure()
    plt.title('Combined Efficient and Naive Eigenvalues')
    plt.xlabel('$l_{m}$ eigenvalue')
    plt.ylabel('Real Value')
    plt.plot(l_naive, label="Naive implementation")
    plt.plot(l_efficient, label="Efficient implementation")
    plt.legend()
    plt.savefig('results/q1/eigenvalues_efficient_naive.png',
                format='png', transparent=True)

    plt.close()


if __name__ == '__main__':
    main()
