# For loading the mat data
from pre_process import load_data
# For calculatuons
import numpy as np

# Import matplotlib
#import matplotlib.pyplot as plt

# Import post process analysing methods
from sklearn.metrics import precision_score

from kNN_manhattan import analyse_KNN_manhattan
from kNN_euclidian import analyse_KNN_euclidian
from kNN_improved_feature_preselection import analyse_KNN_feature_preselection
from kNN_improved_feature_preselection_PCA import analyse_KNN_feature_preselection_PCA
from kNN_cosine import analyse_KNN_cosine
from kNN_improved_nn import analyse_KNN_NN



if __name__ == '__main__':
    k = 10
    #methods = ["Manhattan Distance", "Euclidian Distance", "Cosine"]
    methods = ["Neural Network"]
    results = {}
    for method in methods:
        labels = errors= tops =  true_labels = None
        if method == "Manhattan Distance":
            labels,errors, tops, true_labels = analyse_KNN_manhattan()
        elif method == "Euclidian Distance":
            labels,errors, tops, true_labels = analyse_KNN_euclidian()
        elif method == "Feature pre-selection":
            labels,errors, tops, true_labels = analyse_KNN_feature_preselection()
        elif method == "Feature pre-selection PCA":
            labels,errors, tops, true_labels = analyse_KNN_feature_preselection_PCA()
        elif method == "Cosine":
            labels,errors, tops, true_labels = analyse_KNN_cosine()
        elif method == "Neural Network":
            labels,errors, tops, true_labels = analyse_KNN_NN()
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
        _, errors, _ = results[method]
        plt.plot(X, errors, label=method)
    plt.title("k-NN error")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("results/kNN_errors.png")
    plt.close()

    for method in methods:
        _, _, tops = results[method]
        plt.plot(X, tops, label=method)
    plt.title("k-NN error")
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
