# For loading the mat data
from pre_process import load_data
# For calculatuons
import numpy as np
# Import the kNN library
from sklearn import neighbors
# Import matplotlib
import matplotlib.pyplot as plt
# Import post process analysing methods
from post_process import calculate_scores, plot_confusion_matrix

from sklearn.metrics import precision_score
from sklearn.preprocessing import label_binarize


from tqdm import tqdm

from collections import Counter



def analyse_KNN(k=10):
    """
    Analyse and collect all the different results
    with respect to different kNNs tests
    """

    # Define all the different methods that we are going to try
    methods = ["Manhattan Distance", "Euclidian Distance"]
    results = {}
    for method in methods:
        all_data = load_data()

        training_data = all_data[0]
        query_data = all_data[1]
        gallery_data = all_data[2]

        query_labels = query_data[1]
        training_labels = training_data[1]
        gallery_labels = gallery_data[1]

        query_features = query_data[0]
        training_features = training_data[0]
        gallery_features = gallery_data[0]
#
        query_camIds = query_data[2]
        training_camIds = training_data[2]
        gallery_camIds = gallery_data[2]

        errors = [0]*k
        labels= [None]*k
        tops = [0]*k

        for i in tqdm(range(len(query_features))):
            query = query_features[i,:]
            query_label = query_labels[i]
            query_camId = query_camIds[i]

            selected_gallery_features = []
            selected_gallery_labels = []
            for j in range(len(gallery_features)):
                if not (gallery_camIds[j]==query_camId and gallery_labels[j]==query_label):
                    selected_gallery_features.append(gallery_features[j])
                    selected_gallery_labels.append(gallery_labels[j])

            selected_gallery_features = np.array(selected_gallery_features)
            selected_gallery_labels = np.array(selected_gallery_labels)
            clf = None
            if method == "Manhattan Distance":
                clf = neighbors.KNeighborsClassifier(k,p=1, weights="uniform")
            elif method == "Euclidian Distance":
                clf = neighbors.KNeighborsClassifier(k,p=2, weights="uniform")

            clf.fit(selected_gallery_features, selected_gallery_labels)
            predicted_neighbors = clf.kneighbors(query.reshape(1, -1), return_distance=False)
            predicted_labels = [selected_gallery_labels[l] for l in predicted_neighbors]

            for i in range(len(errors)):
                rank = predicted_labels[0][:i+1]
                b = Counter(rank)
                label = b.most_common(1)[0][0]

                if labels[i] is None:
                    labels[i] = [label]
                else:
                    labels[i].append(label)
                if query_label not in rank:
                    tops[i]+=1

                if label!=query_label:
                    errors[i]+=1


        for i in range(len(errors)):
            errors[i]/=len(query_features)
            tops[i]/=len(query_features)

        results[method] = [labels,errors,tops]

    return methods, results, query_labels

if __name__ == '__main__':
    k = 10
    methods, results, true_labels = analyse_KNN(k)


    for method in methods:
        labels, errors, tops = results[method]
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
