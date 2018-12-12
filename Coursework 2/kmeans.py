# For loading the mat data
from pre_process import load_data
# For calculatuons
import numpy as np
# Import the kNN library
from sklearn.cluster import KMeans
# Import matplotlib
import matplotlib.pyplot as plt
# For the progress bar
from tqdm import tqdm
# For the majority vote counter
from collections import Counter

def analyse_KMeans():
    """
    Analyse and collect all the different results
    with respect to different KMeans

    Note that k is initalized as the number of classes in the test set otherwise
    it would not make sense to do classification at all
    """
    results = {}

    # Split and load the data
    all_data = load_data()

    query_data = all_data[1]
    gallery_data = all_data[2]

    query_labels = query_data[1]
    gallery_labels = gallery_data[1]

    query_features = query_data[0]
    gallery_features = gallery_data[0]

    query_camIds = query_data[2]
    gallery_camIds = gallery_data[2]

    error = 0
    labels = []

    selected_gallery_features = []
    selected_gallery_labels = []

    print("Pre-processing data...")
    for i in tqdm(range(len(query_features))):
        query = query_features[i,:]
        query_label = query_labels[i]
        query_camId = query_camIds[i]
        j = 0
        while j < len(gallery_features):
            if (gallery_camIds[j]==query_camId and gallery_labels[j]==query_label):
                gallery_features = np.delete(gallery_features,j, axis=0)
                gallery_labels = np.delete(gallery_labels, j, axis=0)
                gallery_camIds = np.delete(gallery_camIds, j, axis=0)
            j+=1

    selected_gallery_features = gallery_features
    selected_gallery_labels = gallery_labels

    print("Training classifier...")
    clf = KMeans(max_iter = 100, random_state=1, n_clusters=len(set(selected_gallery_labels)), verbose = True)
    cluster_centers = clf.fit_predict(selected_gallery_features, selected_gallery_labels)

    print("Testing classifier...")
    for i in tqdm(range(len(query_features))):
        query = query_features[i,:]
        query_label = query_labels[i]
        query_camId = query_camIds[i]

        predicted_cluster_center = clf.predict(query.reshape(1, -1))

        predicted_points = []
        predicted_labels = []
        predicted_distances = []

        for i in range(len(cluster_centers)):
            if cluster_centers[i] == predicted_cluster_center:
                predicted_points.append(selected_gallery_features[i])
                predicted_labels.append(selected_gallery_labels[i])

        for i in range(len(predicted_points)):
            distance = np.linalg.norm(clf.cluster_centers_[predicted_cluster_center,:].flatten()-predicted_points[i])
            predicted_distances.append(distance)


        predicted_distances, predicted_labels = zip(*sorted(zip(predicted_distances, predicted_labels)))

        b = Counter(predicted_labels)
        top_label = b.most_common(1)[0][0]
        labels.append(top_label)

        if top_label!=query_label:
            error+=1

    error/=len(query_labels)
    print("Error {}".format(error))
    return [labels,error]

if __name__ == '__main__':
    print(analyse_KMeans())
