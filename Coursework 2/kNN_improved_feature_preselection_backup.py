# For loading the mat data
from pre_process import load_data
# For calculatuons
import numpy as np
# Import the kNN library
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
# Import matplotlib
import matplotlib.pyplot as plt
# Import post process analysing methods
from sklearn.metrics import precision_score
from sklearn.neighbors import DistanceMetric

#from lda import LDA

from tqdm import tqdm

from collections import Counter


def analyse_KNN(k=10):
    """
    Analyse and collect all the different results
    with respect to different kNNs tests

    Parameters
    ----------
    k: int
        How many neighbeours should we consider

    Returns
    -------
    methods: list
        Which methods were considered
    results: list of lists
        Measured results which are going to be later analysed
    true_labels: list
        True test labels
    """

    # Define all the different methods that we are going to try
    methods = ["Cosine"]# ,"Manhattan Distance", "Euclidian Distance"]
    results = {}

    all_data = load_data()
    training_data = all_data[0]

    training_labels = training_data[1]
    training_features = training_data[0]
    training_camIds = training_data[2]

    # Initialize LDA-PCA decomposition
    #pca = PCA(100,whiten=True)
    #training_features = pca.fit_transform(training_features,training_labels)
    #print(training_features.shape)

    #lda = LDA(n_components=200)
    #lda.fit(training_features,training_labels)



    # Fit to the training subspace
    print("Finding W for PCA-LDA transform...")
    #lda.fit()

    # Run the classifier for each method
    for method in methods:
        query_data = all_data[1]
        gallery_data = all_data[2]

        query_labels = query_data[1]
        gallery_labels = gallery_data[1]

        query_features = query_data[0]
        gallery_features = gallery_data[0]

        # If the classifier should be advanced we need to change
        # both the query features as well as gallery features
        if method == "PCA-LDA":
            #gallery_features = pca.transform(gallery_features)
            #query_features = pca.transform(query_features)
            gallery_features = lda.transform(gallery_features)
            query_features = lda.transform(query_features)

        query_camIds = query_data[2]
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
                clf = neighbors.KNeighborsClassifier(k,p=1, weights="uniform",n_jobs= -1)

            elif method == "Euclidian Distance":
                clf = neighbors.KNeighborsClassifier(k,p=2, weights="uniform",n_jobs= -1)

            elif method == "Cosine":
                def w(distances, sigma=0.1):
                    return np.exp(-(distances** 2) / (sigma**2))
                clf = neighbors.KNeighborsClassifier(k, p=2, algorithm="auto",weights=w,n_jobs=-1)


            elif method== "PCA-LDA":
                V = np.cov(selected_gallery_features.T)
                clf = neighbors.KNeighborsClassifier(k, p=2, algorithm="auto",
                                                        weights="distance",
                                                        metric="mahalanobis",
                                                        metric_params={'V': V},
                                                        n_jobs=-1)
                #clf = BaggingClassifier(c, n_estimators=10, random_state=True,
                #                  warm_start=False, n_jobs= -1)

            clf.fit(selected_gallery_features, selected_gallery_labels)
            predicted_neighbors = None
            #if method == "PCA-LDA":
            #    predicted_neighbors = clf.predict(query.reshape(1, -1))
            #else:
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

        results[method] = [labels,errors,tops]

    return methods, results, query_labels
