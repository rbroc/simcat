import pandas as pd
import numpy as np
from .matrix import Matrix
import json
from copy import deepcopy


class Agent:
    """Initializes an agent
    Args:
        agent_name (str): specifies an ID for the agent
        matrix_filename (str): filename of agent's distance matrix
        dict_filename (str): filename of dictionary defining agent's word
            to index mapping (used to compute originality)
        vector_filename (str): filename of agent's vectors
        matrix_kwargs: named arguments for Matrix initialization
    """

    def __init__(
        self, agent_name, matrix_filename, dict_filename=None, vector_filename=None
    ):
        self.name = agent_name or matrix_filename
        self.matrix = Matrix(filename=matrix_filename)
        self.matrix_backup = deepcopy(self.matrix)
        if dict_filename:
            self.position_map = json.load(open(dict_filename, "r"))
        else:
            self.position_map = None
        if vector_filename:
            self.vectors = pd.read_csv(vector_filename, sep="\t", index_col=0)
        else:
            self.vectors = None

    @property
    def model(self):
        return self.matrix.data

    def get_metrics(self, resp_wd, kvals, t):
        """Returns spatial metrics for a turn
        Args:
            resp_wd (str): response word
            kvals (list): k values used to compute metrics such as
                average knn distance
            t (int): stopping threshold for a simulation
        """
        # Distance of response from k-th NN (native neighborhood density)
        knn_dist_resp = [
            (np.sort(self.matrix_backup.data[resp_wd])[k]).round(5) for k in kvals
        ]

        # Distance of response from k-th NN (dynamic neighborhood density)
        knn_dist_resp_current = [
            (np.sort(self.matrix.data[resp_wd])[k]).round(5) for k in kvals
        ]

        # Average distances between each pair of animals (only active words)
        avg_dist_remain = self.matrix.data.mean(axis=0).mean().round(5)

        # Mean distance from neighbors for each word (only active neighbors)
        mean_knn_dists = [
            np.partition(self.matrix.data, kth=k, axis=0)[k, :].mean().round(5)
            for k in kvals
        ]

        # Variance distance from neighbors for each word (only active neighbors)
        var_knn_dists = [
            np.partition(self.matrix.data, kth=k, axis=0)[k, :].std().round(5)
            for k in kvals
        ]

        # Number of sub-threshold neighbors
        resp_neighbors = (self.matrix_backup.data[resp_wd] < t).sum()

        # Number of remaining sub-threshold neighbors
        resp_neighbors_current = (self.matrix.data[resp_wd] < t).sum()

        return [
            *knn_dist_resp,
            *knn_dist_resp_current,
            avg_dist_remain,
            *mean_knn_dists,
            *var_knn_dists,
            resp_neighbors,
            resp_neighbors_current,
        ]

    def speak(self, seed, stopping_rule='distance', pop=True):
        """Picks response word based on cue (seed).
            Returns probability of cue-response association, and response word.
            Also pops response/cue value if pop=True
        Args:
            seed (str): seed word at current turn
            pop (bool): whether to remove the word from memory
        """
        if stopping_rule == 'distance':
            resp_idx = np.argmin(self.matrix.data[seed])
        else:
            resp_idx = np.argmax(self.matrix.data[seed])
        resp_wd = self.matrix.data[seed].index[resp_idx]
        prob = self.listen(seed, resp_wd, pop)
        return round(prob, 5), resp_wd

    def listen(self, seed, resp_wd, pop=True):
        """Listens to response (resp_wd), return probability of response
            given the cue in the agent's space and pops response/cue value
            from agent's memory is pop is true
        Args:
            seed (str): seed word
            resp_wd (str): response
            pop (bool): whether to pop the response from listener's space
        """
        prob = self._return_prob(seed, resp_wd)
        if pop:
            self._pop_words(resp_wd)
        return round(prob, 5)

    def _return_prob(self, seed, resp_wd):
        """Returns association score for seed-resp_wd pair
        Args:
            seed (str): seed word
            resp_wd (str): response word
        """
        return self.matrix.data[seed][resp_wd]

    def _pop_words(self, resp_wd):
        """Pop response word for possible options
        Args:
            resp_wd (str): response word
        """
        self.matrix.data.loc[resp_wd] = np.nan

    def get_vector_var(self, words):
        """Returns average coordinate-based variance for all words named
        during a simulation. Provides an index of how widely the semantic
        space has been explored
        Args:
            words (list): list of words named in the simulation
        """
        return self.vectors.loc[words].values.var(axis=0).mean().round(5)
