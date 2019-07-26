import random as rn
from typing import List, Tuple

import numpy as np
from numpy.random import choice as np_choice


class BinaryFeatureSelectionAntColony:

    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1

        Example:
            ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)
        """
        self.distances = distances
        self.pheromone = np.ones((2, 2))
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.times_taken = np.ones((2, 2))

    def run(self):
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print(shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
        return all_time_shortest_path

    def calculate_feature_amount(self, path: List[Tuple[int, int]]) -> int:
        return len(list(filter(lambda x: x == 1, map(lambda x: x[1], path))))

    def spread_pheronome(self, all_paths, n_best):
        heuristric_sum = 0
        for path, score in all_paths:
            count = self.calculate_feature_amount(path)
            heuristric_sum += score / path
        total_sum = self.decay * heuristric_sum
        for i in range(len(self.pheromone[0]) - 1):
            self.pheromone[0][i] = (1 - self.decay) * self.pheromone[0][i] + total_sum
            self.pheromone[1][i] = (1 - self.decay) * self.pheromone[1][i] + total_sum

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path()
            all_paths.append((path, self.calculate_svm(path)))
        return all_paths

    def gen_path(self):
        path = []
        for i in range(len(self.distances[0]) - 1):
            move: int = self.pick_move(self.pheromone[0][i], self.pheromone[1][i], i)
            self.times_taken[move][i] += 1
            path.append((i, move))
        return path

    def pick_move(self, pheromone_include, pheromone_exclude, i):

        excluded = pheromone_exclude * self.times_taken[0][i]
        included = pheromone_include * self.times_taken[1][i]

        move = np_choice([0, 1], 1, p=[excluded, included])[0]
        return move

    def calculate_svm(self, path) -> int:
        """
        is
        :param path:
        :return: int
        """
        return 10
