from typing import List, Tuple

import numpy as np
from numpy.random import choice as np_choice
from pandas import DataFrame
from sklearn import linear_model


class BinaryFeatureSelectionAntColony:

    def __init__(self, n_ants: int, n_iterations: int, decay: float, data_set: DataFrame, score_weight=1,
                 count_weight=1):
        """
        Args:
            data_set (DataFrame): DataFrame representing input set of features, last column are Y values
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            count_weight (int or float): exponent on count of features
            score_weight (int or float): exponent on score by linear reg
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.score_weight = score_weight
        self.count_weight = count_weight
        self.data_set = data_set
        self.current_iteration: int = 1
        self.times_taken: np.ndarray = np.ones((2, len(self.data_set.columns) - 1))
        self.pheromone: np.ndarray = np.ones((2, len(self.data_set.columns) - 1))
        self.construction_matrix: np.ndarray = np.zeros([2, len(self.data_set.columns) - 1])
        for i in range(len(self.construction_matrix[1])):
            self.construction_matrix[1][i] = 1

    def run(self):
        all_time_shortest_path = ("placeholder", 0)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths)
            shortest_path = max(all_paths, key=lambda x: x[1])
            print(shortest_path)
            if shortest_path[1] > all_time_shortest_path[1] or (
                    shortest_path[1] == all_time_shortest_path[1] and self.calculate_feature_amount(shortest_path[0]) <
                    self.calculate_feature_amount(all_time_shortest_path[0])):
                all_time_shortest_path = shortest_path
            print(self.current_iteration)
            self.current_iteration += 1
        return all_time_shortest_path

    def calculate_feature_amount(self, path: List[Tuple[int, int]]) -> int:
        return len(list(filter(lambda x: x == 1, map(lambda x: x[1], path))))

    def spread_pheronome(self, all_paths: List[Tuple[List[Tuple[int, int]], float]]):
        for path, score in all_paths:
            count = self.calculate_feature_amount(path)
            for i, chosen in path:
                if count == 0:
                    continue
                self.pheromone[chosen][i] = (1 - self.decay) * self.pheromone[chosen][i] + (
                        score / count ** self.count_weight) * self.score_weight
                self.times_taken[chosen][i] += 1

    def gen_all_paths(self) -> List[Tuple[List[Tuple[int, int]], float]]:
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path()
            all_paths.append((path, self.calculate_svm(path)))
        return all_paths

    def gen_path(self) -> List[Tuple[int, int]]:
        path = []
        for i in range(len(self.construction_matrix[0])):
            move: int = self.pick_move(self.pheromone[0][i], self.pheromone[1][i], i)
            path.append((i, move))
        return path

    def pick_move(self, pheromone_exclude: np.ndarray, pheromone_include: np.ndarray, i: int) -> int:

        excluded = pheromone_exclude * (self.times_taken[0][i] / (self.n_ants * self.current_iteration))
        included = pheromone_include * (self.times_taken[1][i] / (self.n_ants * self.current_iteration))

        rows = np.array([excluded, included]) / (excluded + included)

        move = np_choice([0, 1], 1, p=rows)[0]
        return move

    def calculate_svm(self, path: List[Tuple[int, int]]) -> float:
        chosen_features = []
        for i, chosen in path:
            if chosen:
                chosen_features.append(i)
        xs = []
        for i in range(len(self.data_set.iloc[:, 0])):
            temp = []
            for f_id in chosen_features:
                temp.append(self.data_set.iloc[i, f_id])
            xs.append(temp)
        xs = np.asarray(xs)
        ys = self.data_set.iloc[:, len(self.data_set.iloc[0, :]) - 1].values
        regr = linear_model.LinearRegression()
        try:
            regr.fit(xs, ys)
            return regr.score(xs, ys)
        except Exception as e:
            print(e)
            return 0
