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
