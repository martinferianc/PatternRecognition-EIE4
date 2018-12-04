# For loading the mat data
from pre_process import load_data, select_features
# For calculatuons
import numpy as np
# Import the kNN library
from sklearn import neighbors

#from lda import LDA

from tqdm import tqdm

from collections import Counter


def analyse_KNN_euclidian(k=10):
    """
    Analyse and collect all the different results
    with respect to different kNNs tests

    Parameters
    ----------
    k: int
        How many neighbeours should we consider

    Returns
    -------
    results: list of lists
        Measured results which are going to be later analysed
    true_labels: list
        True test labels
    """

    all_data = load_data()
    training_data = all_data[0]

    training_labels = training_data[1]
    training_features = training_data[0]
    training_camIds = training_data[2]

    query_data = all_data[1]
    gallery_data = all_data[2]

    query_labels = query_data[1]
    gallery_labels = gallery_data[1]

    query_features = query_data[0]
    gallery_features = gallery_data[0]

    query_camIds = query_data[2]
    gallery_camIds = gallery_data[2]

    errors = [0]*k
    labels= [None]*k
    tops = [0]*k

    for i in tqdm(range(len(query_features))):
        query = query_features[i,:]
        query_label = query_labels[i]
        query_camId = query_camIds[i]

        selected_gallery_features, selected_gallery_labels = select_features(gallery_camIds, query_camId, gallery_labels, query_label, gallery_features)

        clf = neighbors.KNeighborsClassifier(k,p=2, weights="uniform",n_jobs= -1)

        clf.fit(selected_gallery_features, selected_gallery_labels)

        predicted_neighbors = clf.kneighbors(query.reshape(1, -1), return_distance=False)

        predicted_labels = [selected_gallery_labels[l] for l in predicted_neighbors]
        for i in range(len(predicted_labels[0])):
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

    return labels,errors,tops, query_labels
