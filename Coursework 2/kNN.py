# For calculatuons
import numpy as np

# Import matplotlib
import matplotlib.pyplot as plt

# Import post process analysing methods
from sklearn.metrics import precision_score
from post_process import plot_confusion_matrix

# All the different scripts that we have and the different methods from which we collect the results
from kNN_manhattan import analyse_KNN_manhattan
from kNN_euclidean import analyse_KNN_euclidean
from kNN_improved_RCA_NCA import analyse_KNN_RCA_NCA
from kNN_improved_PCA import analyse_KNN_PCA
from kNN_improved_cosine import analyse_KNN_cosine
from kNN_improved_NN import analyse_KNN_NN

if __name__ == '__main__':
    k = 10
    recollect_results = False
    methods = ["Manhattan Distance", "Euclidian Distance", "Cosine", "RCA & NCA", "Kernel PCA", "Neural Network"]
    results = {}
    true_labels = None
    for method in methods:
        labels = errors= tops = None
        if recollect_results:
            if method == "Manhattan Distance":
                labels, errors, tops, true_labels = analyse_KNN_manhattan()
            elif method == "Euclidian Distance":
                labels,errors, tops, true_labels = analyse_KNN_euclidian()
            elif method == "Kernel PCA":
                labels,errors, tops, true_labels = analyse_KNN_PCA()
            elif method == "RCA & NCA":
                labels,errors, tops, true_labels = analyse_KNN_RCA_NCA()
            elif method == "Cosine":
                labels,errors, tops, true_labels = analyse_KNN_cosine()
            elif method == "Neural Network":
                labels,errors, tops, true_labels = analyse_KNN_NN()
            np.save("results/{}_labels".format(method),labels)
            np.save("results/{}_errors".format(method),errors)
            np.save("results/{}_tops".format(method),tops)
            np.save("results/true_labels".format(method),true_labels)
        else:
            labels = np.load("results/{}_labels.npy".format(method))
            errors = np.load("results/{}_errors.npy".format(method))
            tops = np.load("results/{}_tops.npy".format(method))
            true_labels = np.load("results/true_labels.npy".format(method))
        results[method] = [labels,errors, tops]
        mAPs = []

        for i in range(len(errors)):
            predicted_labels = np.array(labels[i])
            true_labels = np.array(true_labels)
            p_score =  precision_score(true_labels, predicted_labels, average=None)
            mAP = np.mean(p_score)
            mAPs.append(mAP)

        X = list(range(1,k+1))
        plt.plot(X, errors)
        plt.title("k-NN error for {}".format(method))
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.savefig("results/kNN_error_{}.png".format(method))
        plt.close()

        print("k-NN error for {}: {}".format(method,errors))

        plt.plot(X, mAPs)
        plt.title("k-NN mAP for {}".format(method))
        plt.xlabel("k")
        plt.ylabel("mAP")
        plt.savefig("results/kNN_mAP_{}.png".format(method))
        plt.close()

        print("k-NN mAPs for {}: {}".format(method,mAPs))

        plt.plot(X, tops)
        plt.title("k-NN error for {}".format(method))
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.savefig("results/kNN_tops_{}.png".format(method))
        plt.close()

        print("k-NN tops for {}: {}".format(method,tops))

    X = list(range(1,k+1))
    for method in methods:
        labels, _, _= results[method]
        plot_confusion_matrix(true_labels, labels[0,:].T, "results/kNN_CM_{}".format(method),
                                  normalize=True,
                                  title=method,
                                  cmap=plt.cm.Blues)

    for method in methods:
        _, errors, _ = results[method]
        plt.plot(X, errors, label=method)
    plt.title("k-NN Error")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("results/kNN_errors.png")
    plt.close()

    for method in methods:
        _, _, tops = results[method]
        plt.plot(X, tops, label=method)
    plt.title("k-NN Rank Error")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("results/kNN_tops.png")
    plt.close()

    for method in methods:
        labels, errors, tops = results[method]
        mAPs = []
        for i in range(len(errors)):
            predicted_labels = np.array(labels[i])
            true_labels = np.array(true_labels)
            p_score =  precision_score(true_labels, predicted_labels, average=None)
            mAP = np.mean(p_score)
            mAPs.append(mAP)
        plt.plot(X, mAPs, label=method)
    plt.title("k-NN mAPs")
    plt.xlabel("k")
    plt.ylabel("mAP")
    plt.legend()
    plt.savefig("results/kNN_mAPs.png")
    plt.close()
