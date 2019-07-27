import random as rn
from typing import List, Tuple

import numpy as np
from numpy.random import choice as np_choice
from sklearn import linear_model


class BinaryFeatureSelectionAntColony:

    def __init__(self, construction_matrix, n_ants, n_iterations, decay, data_set, alpha=1):
        """
        Args:
            construction_matrix (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1

        Example:
            ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)
        """
        self.construction_matrix = construction_matrix
        self.pheromone = np.ones((2, 2))
        self.all_inds = range(len(construction_matrix))
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.times_taken = np.ones((2, 2))
        self.data_set = data_set
        self.current_iteration = 1

    def run(self):
        all_time_shortest_path = ("placeholder", 0)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths)
            shortest_path = max(all_paths, key=lambda x: x[1])
            print(shortest_path)
            if shortest_path[1] > all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.current_iteration += 1
        return all_time_shortest_path

    def calculate_feature_amount(self, path: List[Tuple[int, int]]) -> int:
        return len(list(filter(lambda x: x == 1, map(lambda x: x[1], path))))

    def spread_pheronome(self, all_paths):
        # heuristric_sum = 0
        # for path, score in all_paths:
        #     count = self.calculate_feature_amount(path)
        #     try:
        #         heuristric_sum += score / count
        #     except ZeroDivisionError:
        #         pass
        # total_sum = (1 - self.decay) * heuristric_sum
        for path, score in all_paths:
            count = self.calculate_feature_amount(path)
            for i, chosen in path:
                if count == 0:
                    continue
                self.pheromone[chosen][i] = (1 - self.decay) * self.pheromone[chosen][i] + (score / count) * self.alpha
                self.times_taken[chosen][i] += 1
        #
        # for i in range(len(self.pheromone[0])):
        #     self.pheromone[0][i] = (1 - self.decay) * self.pheromone[0][i] + total_sum
        #     self.pheromone[1][i] = (1 - self.decay) * self.pheromone[1][i] + total_sum

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path()
            all_paths.append((path, self.calculate_svm(path)))
        return all_paths

    def gen_path(self):
        path = []
        for i in range(len(self.construction_matrix[0])):
            move: int = self.pick_move(self.pheromone[0][i], self.pheromone[1][i], i)
            path.append((i, move))
        return path

    def pick_move(self, pheromone_exclude, pheromone_include, i):

        excluded = pheromone_exclude * (self.times_taken[0][i] / (self.n_ants * self.current_iteration))
        included = pheromone_include * (self.times_taken[1][i] / (self.n_ants * self.current_iteration))

        rows = np.array([excluded, included]) / (excluded + included)

        move = np_choice([0, 1], 1, p=rows)[0]
        return move

    def calculate_svm(self, path) -> float:
        chosen_features = []
        for i, chosen in path:
            if chosen:
                chosen_features.append(i)
        xs = []
        for i in range(len(self.data_set[0])):
            temp = []
            for f_id in chosen_features:
                temp.append(self.data_set[f_id][i][0])
            xs.append(temp)
        ys = self.data_set[len(self.data_set) - 1]
        regr = linear_model.LinearRegression()
        try:
            regr.fit(xs, ys)
            return regr.score(xs, ys)
        except Exception as e:
            print(e)
            return 0
