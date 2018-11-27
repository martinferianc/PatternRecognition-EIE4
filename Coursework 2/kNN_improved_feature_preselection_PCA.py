# For loading the mat data
from pre_process import load_data, select_features
# For calculatuons
import numpy as np
# Import the kNN library
from sklearn import neighbors

# Import matplotlib
import matplotlib.pyplot as plt

# Import post process analysing methods
from learn_distance_metric import find_matrices
from sklearn.preprocessing import normalize
from sklearn.decomposition import KernelPCA

import metric_learn

from tqdm import tqdm

from collections import Counter

"""
def metric(x,y, **kwargs):
    U = kwargs["metric_params"]["U"]
    A = kwargs["metric_params"]["A"]
    s1 = np.matmul(A.T, x-y)
    s2 = np.matmul(x-y,A)
    s3 = np.matul(np.matmul(s1.T,A),s_2)
    return np.sqrt(s3)
"""
def weight(x, sigma=0.1):
    return np.exp(-(x** 2) / 2*(sigma**2))

def vote(x, weights):
    label = -1
    best_score = -float('inf')
    for i in range(len(x)):
        score = 0
        for j in range(len(x)):
            if x[i] == x[j]:
                score+=weights[j]
        if score> best_score:
            label = x[i]
            best_score = score
    return label


def analyse_KNN_feature_preselection_PCA(k=10):
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

    all_data = load_data(False)
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
    query_features = normalize(query_features, axis=1)
    training_features = normalize(training_features, axis=1)
    gallery_features = normalize(gallery_features, axis=1)
    pca = KernelPCA(n_components=500, kernel="rbf", n_jobs=-1)
    pca.fit(training_features)

    query_features      = pca.transform(query_features)
    training_features   = pca.transform(training_features)
    gallery_features    = pca.transform(gallery_features)

    for i in tqdm(range(len(query_features))):
        query = query_features[i,:]
        query_label = query_labels[i]
        query_camId = query_camIds[i]

        selected_gallery_features, selected_gallery_labels = select_features(gallery_camIds, query_camId, gallery_labels, query_label, gallery_features)

        clf = neighbors.KNeighborsClassifier(k,algorithm="brute", metric="euclidean")
        #clf = neighbors.KNeighborsClassifier(k,algorithm='brute',metric=metric,
        #                                    metric_params={"A": A_s[query_label], "U": U_s[query_label]})

        clf.fit(selected_gallery_features, selected_gallery_labels)
        distances, predicted_neighbors = clf.kneighbors(query.reshape(1, -1), return_distance=True)
        predicted_labels = np.array([selected_gallery_labels[l] for l in predicted_neighbors]).flatten()

        weighted_distances = weight(distances).flatten()

        for j in range(len(predicted_labels)):
            rank = predicted_labels[:j+1]
            rank_weights = weighted_distances[:j+1]
            label = vote(rank, rank_weights)

            if labels[j] is None:
                labels[j] = [label]
            else:
                labels[j].append(label)
            if query_label not in rank:
                tops[j]+=1

            if label!=query_label:
                errors[j]+=1
    for i in range(len(errors)):
        errors[i]/=len(query_features)
        tops[i]/=len(query_features)

    return labels,errors,tops, query_labels
