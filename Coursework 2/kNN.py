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

def find_best_K(data,type="naive"):
    """
    Finding the best K for each of the methods
    shown in the lecture notes

    Parameters
    ----------
    data: list
        Data which can be training, validation or
        test including the labels
    type: str
        Which implementation of kNN should we look
        k for
    Returns
    -------
    results: dict
        * A dictionary containing the errors for the weighted
        as well as unoformly evaluated distances
    """
    training_data = data[0][0]
    training_labels = data[0][1]

    validation_data = data[1][0]
    validation_labels = data[1][1]

    types = ["uniform", "distance"]

    results = {}
    for weights in types:
        weight_errors = []
        clf = None
        if type == "naive":
            clf = neighbors.KNeighborsClassifier(1, weights=weights)
        else:
            raise Exception("Wrong method!")
        clf.fit(training_data, training_labels)
        print("Model fitted...")

        for k in range(1,10):
            print("Finding k={} for weight type={}".format(k,weights))

            clf.neighbors = k

            # Predict
            predicted_labels = clf.predict(validation_data)
            error = 0
            for i in range(len(predicted_labels)):
                if predicted_labels[i]!=validation_labels[i]:
                    error+=1
            error/=len(validation_labels)

            weight_errors.append(error)

        results[weights] = np.array(weight_errors)

    return results

def test(data, k, weighting, type="naive"):
    """
    Evaluating the best found k
    on the test data rogether with the
    type of weighting as well as type
    of kNN

    Parameters
    ----------
    data: list
        Data which can be training, validation or
        test including the labels
    k: int
        Best found k found for this method/type
    type: str
        Which implementation of kNN should we look
        k for
    Returns
    -------
    error: float
        * Error computed on the test set
    test_labels: list
        * Labels needed for the further analysis
    predicted_labels: list
        * Labels needed for the further analysis
    """
    training_data = data[0][0]
    training_labels = data[0][1]

    test_data = data[2][0]
    test_labels = data[2][1]

    results = {}
    clf = None
    if type == "naive":
        clf = neighbors.KNeighborsClassifier(k, weights=weighting)
    else:
        raise Exception("Wrong method!")

    clf.fit(training_data, training_labels)
    print("Model fitted...")

    # Predict
    predicted_labels = clf.predict(test_data)
    error = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]!=test_labels[i]:
            error+=1
    error/=len(test_labels)

    return error, test_labels, predicted_labels

def analyse_KNN():
    """
    Analyse and collect all the different results
    with respect to different kNNs tests
    """

    # Define all the different methods that we are going to try
    methods = ["naive"]
    for method in methods:
        all_data = load_data()
        # Find the best results for all different methods and types
        results = find_best_K(all_data, type=method)

        # Generate corresponding plots
        x = list(range(1,len(results["uniform"])+1))
        plt.figure()
        plt.title('Error for uniformly weighted {} kNN classifier'.format(method))
        plt.xlabel('k Neighbours')
        plt.ylabel('Error')
        plt.axis([1, 9,0,1])
        plt.plot(x,results["uniform"],"bo")
        plt.savefig('results/kNN/uniform_{}_kNN.png'.format(method),
                        format='png', transparent=True)

        plt.figure()
        plt.title('Error for distance weighted {} kNN classifier'.format(method))
        plt.xlabel('k Neighbours')
        plt.ylabel('Error')
        plt.axis([1, 9,0,1])
        plt.plot(x,results["distance"],"bo")
        plt.savefig('results/kNN/weighted_{}_kNN.png'.format(method),
                            format='png', transparent=True)

        types = ["uniform", "distance"]

        for t in types:
            best_k = np.argmin(results[t])+1
            error, test_labels, predicted_labels = test(all_data, best_k, t)
            calculate_scores(test_labels, predicted_labels, "results/kNN/{}_{}_{}_scores".format(t,best_k,method))
            plot_confusion_matrix(test_labels, predicted_labels, "results/kNN/{}_{}_{}_CM".format(t,best_k,method))
            plot_confusion_matrix(test_labels, predicted_labels, True, "results/kNN/{}_{}_{}_CM_normalized".format(t,best_k,method))


if __name__ == '__main__':
    analyse_KNN()
