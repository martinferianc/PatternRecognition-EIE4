# For loading the mat data
from pre_process import load_data, select_features
# For calculatuons
import numpy as np
# Import the kNN library
from sklearn import neighbors

# Import matplotlib
import matplotlib.pyplot as plt

# Import post process analysing methods
from sklearn.feature_selection import VarianceThreshold

from lda import *

from tqdm import tqdm

from collections import Counter

def weight(x, sigma=0.01):
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


def analyse_KNN_feature_preselection(k=10):
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

#    selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
#    training_features = selector.fit_transform(training_features, training_labels)
#    gallery_features = selector.transform(gallery_features)
#    query_features = selector.transform(query_features)

#    lda_W = LDA(training_features.T, training_labels)

#    training_features = LDA_transform(lda_W,training_features.T)
#    gallery_features = LDA_transform(lda_W,gallery_features.T)
#    query_features = LDA_transform(lda_W,query_features.T)


    for i in tqdm(range(len(query_features))):
        query = query_features[i,:]
        query_label = query_labels[i]
        query_camId = query_camIds[i]

        selected_gallery_features, selected_gallery_labels = select_features(gallery_camIds, query_camId, gallery_labels, query_label, gallery_features)


        clf = neighbors.KNeighborsClassifier(k,algorithm='brute',metric='seuclidean')

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
        print(errors[0]/(i+1))
    for i in range(len(errors)):
        errors[i]/=len(query_features)
        tops[i]/=len(query_features)

    return labels,errors,tops, query_labels
