import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm
import copy

from pre_process_raw import load_data
from lda import LDA

def main():

    # Load dataset
    dataset = load_data()

    ######################
    # PCA-LDA EVALUATION #
    ######################

    # Evaluate for different M_pca
    M_pca = [5,20,50,100]
    M_lda = [5,20,50,100]

    for m_pca in M_pca:
        for m_lda in M_lda:
            # Setup
            lda = LDA()
            lda.dataset = copy.deepcopy(dataset)
            lda.run_setup()

            # Set hyper parameters
            lda.M_pca = m_pca
            lda.M_lda = m_lda

            # Run
            lda.run_pca_lda()
            lda.run_nn_classifier()


    # TODO:
    # - print fisherfaces
    # - comittee machine
    #   - perform bagging
    #   - parameter randomisation (w_pca and w_lda vectors)

    ###################
    # PCA-LDA BAGGING #
    ###################

    # Load dataset
    dataset = load_data()

    # Number of machines
    NUM_MACHINES = 5

    # Machine Parameters
    M_pca = 50
    M_lda = 50
    sample_size = 20

    machine = [LDA() for i in range(NUM_MACHINES)]

    for i in range(NUM_MACHINES):
        # Randomly sample training data TODO try stratified and un-stratified
        sample_index = [0] # TODO generate random list of indicies

        # assign dataset for machine
        machine[i].dataset['train_x'] = copy.deepcopy(dataset['train_x'][:,sample_index])
        machine[i].dataset['train_y'] = copy.deepcopy(dataset['train_y'][:,sample_index])

        machine[i].dataset['test_x'] = copy.deepcopy(dataset['test_x'])
        machine[i].dataset['test_y'] = copy.deepcopy(dataset['test_y'])

        # Setup each machine
        machine[i].split_classes()
        machine[i].get_mean()
        machine[i].M_pca = M_pca
        machine[i].M_lda = M_lda

if __name__ == '__main__':
    main()
