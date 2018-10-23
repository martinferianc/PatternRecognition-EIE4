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

from tqdm import tqdm



def analyse_KNN(k=5):
    """
    Analyse and collect all the different results
    with respect to different kNNs tests
    """

    # Define all the different methods that we are going to try
    methods = ["naive"]
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

        query_camIds = query_data[2]
        training_camIds = training_data[2]
        gallery_camIds = gallery_data[2]

        error = 0

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

            clf = neighbors.KNeighborsClassifier(1, weights="uniform")
            clf.fit(selected_gallery_features, selected_gallery_labels)

            predicted_label = clf.predict(query.reshape(1, -1))

            if predicted_label!=query_label:
                error+=1
            print("Error: ", error/(i+1))
        print("Error: ", error/len(query_labels))


if __name__ == '__main__':
    analyse_KNN()
