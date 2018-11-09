import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm
import copy

from pre_process import load_data
from lda import LDA

def main():
    ######################
    # PCA-LDA EVALUATION #
    ######################

    # Evaluate for different M_pca
    M_pca = [5,20,50,100]
    for m in M_pca:
        pass # TODO: run pca-lda (fixed M_lda)

    # Evaluate for different M_lda
    M_lda = [5,20,50,100]
    for m in M_lda:
        pass # TODO: run pca-lda (fixed M_pca)


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
