import numpy as np
# For file manipulation and locating
import os
# For the progress bar
from tqdm import tqdm
# To create a deep copy of the data
import copy
# To load the pre-processed and split data
from pre_process import load_data as ld
# For normalization of the samples
from sklearn.preprocessing import normalize

# We define some constant that we reuse
PROCESSED_DIR = "data/processed/"

def save_data(data, file_path, name):
    """
    Saves the data
    given the name and
    the file path

    Parameters
    ----------
    data: numpy matrix
        Data matrix with features
    file_path: str
        File path where the file should be saved
    name: str
        Specific name of the given file

    """
    np.save(file_path + "{}.npy".format(name),data)

def preprocess(X, Y, size = 100000,lower_bound=0, upper_bound = 7368,samples = 10, same_class=0.4, different = 0.5, penalty = 10, same_class_penalty=1):
    """
    Preprocessed the dataset

    It creates two lists X,Y
    It randomly chooses a sample from the input list
    and then that sample is repeated in total * samples time
    For each repeated sample it finds a portion of
    images corresponding to different labels,
    images corresponding to the same class and
    a certain portion of identities
    based on the class membership a penalty is applied or not

    Parameters
    ----------
    X: numpy array of features
        Numpy array of features from which the pairs are created
    Y: numpy array
        Numpy array of corresponding labels
    Returns
    -------
    X_selected: numpy array
        Numpy array of the first input in the pairs

    Y_selected: numpy array
        Numpy array of the second input in the pairs

    values: numpy array
        Artificially determined distances

    """
    X = normalize(X, axis=1)

    N,F = X.shape

    X_selected = []
    Y_selected = []
    values = []

    C = int(samples*same_class)
    D = int(samples*different)

    selected_i = []

    for i in tqdm(range(int(size/samples))):
        # Randomly select a sample but do not repeat it with respect ot previous samples
        random_i = np.random.randint(lower_bound,upper_bound)
        while random_i in selected_i:
            random_i = np.random.randint(lower_bound,upper_bound)
        selected_i.append(random_i)
        C_counter = 0
        D_counter = 0
        offset = 0

        # Add samples which correspond to different label than the original image
        selected_j = []
        while D_counter<D:
            random_j = np.random.randint(lower_bound,upper_bound)
            while random_j in selected_j:
                random_j = np.random.randint(lower_bound,upper_bound)
            if Y[random_i] != Y[random_j]:
                D_counter+=1
                offset+=1
                X_selected.append(copy.deepcopy(X[random_i]))
                Y_selected.append(copy.deepcopy(X[random_j]))
                values.append(penalty)
                selected_j.append(random_j)

        # Add samples which correspond to the same class
        selected_j = []
        while C_counter<C:
            low = 0
            high = N
            if random_i-10>lower_bound:
                low = random_i-10
            if random_i+10<upper_bound:
                high = random_i+10
            random_j = np.random.randint(lower_bound,upper_bound)
            while random_j in selected_j:
                random_j = np.random.randint(lower_bound,upper_bound)
            if Y[random_i] == Y[random_j] and random_i!=random_j:
                C_counter+=1
                offset +=1
                X_selected.append(copy.deepcopy(X[random_i]))
                Y_selected.append(copy.deepcopy(X[random_j]))
                values.append(same_class_penalty)
                selected_j.append(random_j)

        # Fill in the rest with identities
        while offset < samples:
            X_selected.append(copy.deepcopy(X[random_i]))
            Y_selected.append(copy.deepcopy(X[random_i]))
            offset+=1
            values.append(0)

    indeces = np.random.choice(size, size=size, replace=False)
    X_selected = np.array(X_selected)
    Y_selected = np.array(Y_selected)
    values = np.array(values)

    return [X_selected[indeces], Y_selected[indeces], values[indeces]]

def load_data(retrain=False):
    """
    Load the cached data or call preprocess()
    to generate new data

    Parameters
    ----------
    None

    Returns
    -------
    all_data: list
        * All the data split into lists of [features, labels]

    """
    all_data = ld(False)
    training_data = all_data[0]

    Y = training_data[1]
    X = training_data[0]

    if retrain is True:
        print("Generating new data...")
        X_train, Y_train, values_train = preprocess(X,Y, 40000, 0, 6379,samples = 10, same_class=0.4, different = 0.5, penalty = 1,same_class_penalty=0)
        X_validation, Y_validation, values_validation = preprocess(X,Y, 7500, 6380,samples = 10, same_class=0.2, different = 0.7, penalty = 1, same_class_penalty=0)
        save_data(X_train,PROCESSED_DIR,"training_nn_X")
        save_data(Y_train,PROCESSED_DIR,"training_nn_Y")
        save_data(values_train,PROCESSED_DIR,"training_nn_values")
        save_data(X_validation,PROCESSED_DIR,"validation_nn_X")
        save_data(Y_validation,PROCESSED_DIR,"validation_nn_Y")
        save_data(values_validation,PROCESSED_DIR,"validation_nn_values")
        return [X_train, Y_train, values_train, X_validation, Y_validation, values_validation]
    else:
        print("Loading data...")
        data = []
        data.append(np.load(PROCESSED_DIR + "training_nn_X.npy"))
        data.append(np.load(PROCESSED_DIR + "training_nn_Y.npy"))
        data.append(np.load(PROCESSED_DIR + "training_nn_values.npy"))
        data.append(np.load(PROCESSED_DIR + "validation_nn_X.npy"))
        data.append(np.load(PROCESSED_DIR + "validation_nn_Y.npy"))
        data.append(np.load(PROCESSED_DIR + "validation_nn_values.npy"))
        return data

if __name__ == '__main__':
    load_data(retrain=True)
