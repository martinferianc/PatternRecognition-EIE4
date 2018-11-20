import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import tqdm
import copy
from random import randint, sample
from statistics import mode
import random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from post_process import plot_confusion_matrix


from pre_process_raw import load_data
from lda import LDA

def sample_rnd(train_y,sample_size):
    max_size = train_y.shape[0]
    return [randint(0,max_size-1) for i in range(sample_size)]

def sample_stratified(train_y,sample_size):
    # Set of labels
    labels = np.unique(train_y)

    rnd_indices = []

    # Get indices for each
    for label in labels:
        # Get indices for those labels
        indices = []
        for i in range(train_y.shape[0]):
            indices.append(i) if (train_y[i] == label) else 0
        # sample subset of indices
        rnd_indices.extend(sample(indices,sample_size))
    return rnd_indices

def committe_machine_majority_vote(labels):
    # number of machines
    num_machines = len(labels)

    # number of labels
    num_labels = len(labels[0])

    labels_out = []

    for i in range(num_labels):
        votes = []
        for j in range(num_machines):
            votes.append(labels[j][i])
        labels_out.append(max(set(votes), key = votes.count))

    return labels_out

def committe_machine_average(labels):
    # number of machines
    num_machines = len(labels)

    # number of labels
    num_labels = len(labels[0])

    labels_out = []

    for i in range(num_labels):
        avg = 0
        for j in range(num_machines):
            avg += labels[j][i]
        labels_out.append(int(avg/num_machines))
    return labels_out

def committe_machine_weighted_voting(labels,class_sizes):
    # number of machines
    num_machines = len(labels)

    # number of labels
    num_labels = len(labels[0])

    labels_out = []

    for i in range(num_labels):
        votes = []
        for j in range(num_machines):
            votes.extend([ labels[j][i] ]*class_sizes[j][labels[j][i]])
        labels_out.append(max(set(votes), key = votes.count))

    return labels_out

def random_parameters(M0,M1,max_size=405):
    vec_index = np.arange(M0).tolist()
    vec_index.extend(random.sample(range(M0, max_size), M1))
    return vec_index

def identity_error(labels, labels_correct):
    err = 0
    for i in range(len(labels)):
        if labels[i] != labels_correct[i]:
            err += 1
    #normalise by size of labels
    return err/len(labels)

def main():

    # Load dataset
    dataset = load_data()

    '''

    ##############
    # BASIC TEST #
    ##############

    # Setup
    lda = LDA()
    lda.dataset = copy.deepcopy(dataset)
    lda.run_setup()

    # Set hyper parameters
    lda.M_pca = 150
    lda.M_lda = 40

    # Run
    lda.run_pca_lda()
    lda.run_nn_classifier()




    ######################
    # PCA-LDA EVALUATION #
    ######################

    # Evaluate for different M_pca
    M_pca = np.arange(75,300,10)
    M_lda = np.arange(20,100,10)

    err_results = [ [] for m in M_lda ]
    lda_index = 0

    for m_lda in M_lda:
        for m_pca in M_pca:
            if m_lda > m_pca:
                continue

            # Setup
            lda = LDA()
            lda.dataset = copy.deepcopy(dataset)
            lda.run_setup()

            # Set hyper parameters
            lda.M_pca = m_pca
            lda.M_lda = m_lda

            # Run
            lda.run_pca_lda()
            err,_ = lda.run_nn_classifier()

            print("M PCA: {}, M LDA: {}, ERROR: {}".format(m_pca,m_lda,err))

            err_results[lda_index].append(err)

        lda_index += 1

    fig = plt.figure()
    legends = [ '' for i in range(len(err_results)) ]
    for i in range(len(err_results)):
        legends[i], = plt.plot(M_pca,err_results[i],label='M lda = {}'.format(M_lda[i]))
    plt.legend(handles=legends)
    plt.show()

    '''

    '''
    ###################
    # PCA-LDA BAGGING #
    ###################

    # Number of machines
    NUM_MACHINES = 5

    # Machine Parameters
    M_pca = 100
    M_lda = 50
    sample_size = 300

    machine = [LDA() for i in range(NUM_MACHINES)]
    class_sizes = []

    for i in range(NUM_MACHINES):
        # Randomly sample training data TODO try stratified and un-stratified
        sample_index = sample_rnd(dataset['train_y'],sample_size)
        #sample_index = sample_stratified(dataset['train_y'],sample_size)

        # assign dataset for machine
        machine[i].dataset['train_x'] = copy.deepcopy(dataset['train_x'][:,sample_index])
        machine[i].dataset['train_y'] = copy.deepcopy(dataset['train_y'][sample_index])

        machine[i].dataset['test_x'] = copy.deepcopy(dataset['test_x'])
        machine[i].dataset['test_y'] = copy.deepcopy(dataset['test_y'])

        # Setup each machine
        machine[i].run_setup()
        machine[i].M_pca = M_pca
        machine[i].M_lda = M_lda

        class_sizes.append(machine[i].get_class_sizes())

    # variable to store label results
    labels =  [[] for i in range(NUM_MACHINES)]

    for i in range(NUM_MACHINES):
        machine[i].run_pca_lda()
        _, labels[i] = machine[i].run_nn_classifier()

    # get committee machine output
    labels_out = committe_machine_majority_vote(labels)
    err = identity_error(labels_out,dataset['test_y'])

    print('error(majority voting): ',err)

    # get committee machine output
    labels_out = committe_machine_weighted_voting(labels,class_sizes)
    err = identity_error(labels_out,dataset['test_y'])

    print('error(weighted voting): ',err)

    # get committee machine output (average)
    labels_out = committe_machine_average(labels)
    err = identity_error(labels_out,dataset['test_y'])

    print('error(average): ',err)

    '''


    ###################################
    # PCA-LDA PARAMETER RANDOMISATION #
    ###################################

    # Number of machines
    NUM_MACHINES = 15

    # Machine Parameters
    M0 = 125
    M1 = 25

    #M_pca = 100
    M_lda = 40
    #sample_size = 5

    machine = [LDA() for i in range(NUM_MACHINES)]

    for i in range(NUM_MACHINES):
        # Choose random eigenvectors for PCA
        M_pca = random_parameters(M0,M1,max_size=(len(dataset['train_y'])-1))

        # assign dataset for machine
        machine[i].dataset['train_x'] = copy.deepcopy(dataset['train_x'])
        machine[i].dataset['train_y'] = copy.deepcopy(dataset['train_y'])

        machine[i].dataset['test_x'] = copy.deepcopy(dataset['test_x'])
        machine[i].dataset['test_y'] = copy.deepcopy(dataset['test_y'])

        # Setup each machine
        machine[i].run_setup()
        machine[i].M_pca = M_pca
        machine[i].M_lda = M_lda

    # variable to store label results
    labels =  [[] for i in range(NUM_MACHINES)]

    for i in range(NUM_MACHINES):
        machine[i].run_pca_lda(m_pca_type=1)
        _, labels[i] = machine[i].run_nn_classifier()

    # get committee machine output
    labels_out = committe_machine_majority_vote(labels)
    err = identity_error(labels_out,dataset['test_y'])

    print('error(majority voting): ',err)

    # get committee machine output (average)
    labels_out = committe_machine_average(labels)
    err = identity_error(labels_out,dataset['test_y'])

    print('error(average): ',err)
    plot_confusion_matrix(dataset["test_y"], labels_out, "results/q3/lda_pca_ensemble_classifier_cm",normalize=True)




    ############################
    # ENSEMBLE HYPERPARAMETERS #
    ############################

    # Number of machines
    NUM_MACHINES = 50

    # List of errors
    err = [ [0,0] for i in range(NUM_MACHINES) ]
    err = [
        [0 for i in range(NUM_MACHINES) ],
        [0 for i in range(NUM_MACHINES) ]
    ]

    # HIGH CORRELATION #

    # Machine Parameters
    M0 = 125
    M1 = 25

    #M_pca = 100
    M_lda = 40
    #sample_size = 5

    machine = [LDA() for i in range(NUM_MACHINES)]

    for i in range(NUM_MACHINES):
        # Choose random eigenvectors for PCA
        M_pca = random_parameters(M0,M1,max_size=(len(dataset['train_y'])-1))

        # assign dataset for machine
        machine[i].dataset['train_x'] = copy.deepcopy(dataset['train_x'])
        machine[i].dataset['train_y'] = copy.deepcopy(dataset['train_y'])

        machine[i].dataset['test_x'] = copy.deepcopy(dataset['test_x'])
        machine[i].dataset['test_y'] = copy.deepcopy(dataset['test_y'])

        # Setup each machine
        machine[i].run_setup()
        machine[i].M_pca = M_pca
        machine[i].M_lda = M_lda

    # variable to store label results
    labels =  [[] for i in range(NUM_MACHINES)]

    for i in range(NUM_MACHINES):
        machine[i].run_pca_lda(m_pca_type=1)
        _, labels[i] = machine[i].run_nn_classifier()

    # get committee machine output
    for i in range(NUM_MACHINES):
        labels_out = committe_machine_majority_vote(labels[:(i+1)])
        err[0][i]  = identity_error(labels_out,dataset['test_y'])

    # LOW CORRELATION #

    # Machine Parameters
    M0 = 25
    M1 = 125

    #M_pca = 100
    M_lda = 40
    #sample_size = 5

    machine = [LDA() for i in range(NUM_MACHINES)]

    for i in range(NUM_MACHINES):
        # Choose random eigenvectors for PCA
        M_pca = random_parameters(M0,M1,max_size=(len(dataset['train_y'])-1))

        # assign dataset for machine
        machine[i].dataset['train_x'] = copy.deepcopy(dataset['train_x'])
        machine[i].dataset['train_y'] = copy.deepcopy(dataset['train_y'])

        machine[i].dataset['test_x'] = copy.deepcopy(dataset['test_x'])
        machine[i].dataset['test_y'] = copy.deepcopy(dataset['test_y'])

        # Setup each machine
        machine[i].run_setup()
        machine[i].M_pca = M_pca
        machine[i].M_lda = M_lda

    # variable to store label results
    labels =  [[] for i in range(NUM_MACHINES)]

    for i in range(NUM_MACHINES):
        machine[i].run_pca_lda(m_pca_type=1)
        _, labels[i] = machine[i].run_nn_classifier()

    # get committee machine output
    for i in range(NUM_MACHINES):
        labels_out = committe_machine_majority_vote(labels[:(i+1)])
        err[1][i]  = identity_error(labels_out,dataset['test_y'])

    plt.figure()
    plt.title('Comparison of Different Comittee Machines')
    plt.xlabel('Number of Machines')
    plt.ylabel('Error (%)')
    plt.plot(range(NUM_MACHINES),err[0], label="High Machine Correlation")
    plt.plot(range(NUM_MACHINES),err[1], label="Low Machine Correlation")
    plt.legend()
    plt.savefig('results/q3/num_machines_eval.png',
                format='png', transparent=True)


if __name__ == '__main__':
    main()
