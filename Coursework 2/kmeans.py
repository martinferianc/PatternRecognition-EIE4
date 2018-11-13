# For loading the mat data
from pre_process import load_data
# For calculatuons
import numpy as np
# Import the kNN library
from sklearn.cluster import KMeans
# Import matplotlib
import matplotlib.pyplot as plt
# Import post process analysing methods
from post_process import calculate_scores, plot_confusion_matrix

from tqdm import tqdm

from collections import Counter




def analyse_KMeans():
    """
    Analyse and collect all the different results
    with respect to different kNNs tests
    """

    # Define all the different methods that we are going to try
    methods = ["normal"]
    results = {}
    for method in methods:
        all_data = load_data()

        training_data = all_data[0]
        query_data = all_data[1]
        gallery_data = all_data[2]

        query_labels = query_data[1]#[:100]
        training_labels = training_data[1]#[:100]
        gallery_labels = gallery_data[1]#[:100]

        query_features = query_data[0]#[:100,:]
        training_features = training_data[0]#[:100,:]
        gallery_features = gallery_data[0]#[:100,:]

        query_camIds = query_data[2]#[:100]
        training_camIds = training_data[2]#[:100]
        gallery_camIds = gallery_data[2]#[:100]

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
        clf = None
        if method == "normal":
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
        print("Error for {} method: {}".format(method,error))
        results[method] = [labels,error]


    return methods, results, query_labels


if __name__ == '__main__':
    analyse_KMeans()
