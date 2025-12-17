import numpy as np
from src.train import *

def Viterbi(sample_len, state_graph_len):

    parameters = train_hmm_supervised_with_unk()
    viterbi = np.zeros((state_graph_len, sample_len))
    backpointer = np.zeros((state_graph_len, sample_len))

    for s in range(state_graph_len):
        viterbi[s, 1] = parameters["transition_probabilities"][START_TAG][s] * parameters["emission_probabilities"][s][1]
        backpointer[s, 1] = 0

    for t in range(2, sample_len):
        for s in range(state_graph_len):

            viterbi[s,t] = max()