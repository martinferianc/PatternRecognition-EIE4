import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm
import copy

from pre_process import load_mat, remove_mean, separate_data, compute_eigenvalues_eigenvectors

def sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    p = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues_ = copy.deepcopy(eigenvalues[p])
    eigenvectors_ = copy.deepcopy(np.real(eigenvectors[:,p]))
    return eigenvalues_, eigenvectors_

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
    X_train = dataset[0][0]

    # Compute and remove mean from the training dataset
    mean = X_train.mean(axis=1).reshape(-1,1)


    A = X_train - mean


    #Display the mean face for verification
    plt.figure()
    plt.title('Mean Face')
    mean = mean.reshape((46,56))
    mean = np.rot90(mean,3)
    plt.imshow(mean, cmap="gray")
    plt.savefig("results/q1/mean.png", format="png", transparent=True)

    D, N = X_train.shape
    start = time.time()
    S_naive = copy.deepcopy((1 / N) * np.dot(A, A.T))
    end = time.time()
    print("Calculated S inefficiently, took: {} s".format(end-start))

    start = time.time()
    S_efficient = copy.deepcopy((1/N) * np.dot(A.T, A))
    end = time.time()
    print("Calculated S efficiently, took: {} s".format(end-start))

    start = time.time()
    _l_naive,_u_naive = np.linalg.eig(S_naive)
    end = time.time()
    print("Calculated eigenvectors, eigenvalues efficiently, took: {} s".format(end-start))

    start = time.time()
    _l_efficient,_u_efficient = np.linalg.eig(S_efficient)
    end = time.time()
    print("Calculated eigenvectors, eigenvalues inefficiently, took: {} s".format(end-start))


    _u_efficient = copy.deepcopy(np.dot(A, copy.deepcopy(_u_efficient)))
    indexes = np.argsort(np.abs(_l_naive))[::-1]
    u_naive = np.real(_u_naive[:, indexes])
    l_naive = _l_naive[indexes]

    _indexes = np.argsort(np.abs(_l_efficient))[::-1]
    u_efficient = np.real(_u_efficient[:, _indexes])
    l_efficient = _l_efficient[_indexes]



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

    f, axarr = plt.subplots(2,3)
    for i in range(3):
        img = np.real(u_naive[:,i].reshape((46,56)))
        img = np.rot90(img,3)
        axarr[0,i].imshow(img, cmap=plt.cm.Greys)
        axarr[0,i].set_title("Eigenface %d" %i)
    for i in range(3):
        img = np.real(u_efficient[:,i].reshape((46,56)))
        img = np.rot90(img,3)
        axarr[1,i].imshow(img, cmap=plt.cm.Greys)

    plt.savefig("results/q1/eigenfaces.png", format="png", transparent=True)

    plt.close()


if __name__ == '__main__':
    main()
