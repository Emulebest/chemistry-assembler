import unittest
from typing import List

import numpy as np

from aco.ant_colony import BinaryFeatureSelectionAntColony


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.featureA = np.array([[1], [2], [3], [4]])
        self.featureB = np.array([[100], [107], [112], [108]])
        self.result = np.array([[11], [19], [31], [41]])

    def test_ACO(self):
        construction_matrix: List[List[int]] = [[0, 0],
                                                [1, 1]]
        data_set = [self.featureA, self.featureB, self.result]
        ant_colony = BinaryFeatureSelectionAntColony(construction_matrix, 4, 2, 0.05, data_set, 2)
        result_path = ant_colony.run()
        self.assertGreaterEqual(result_path[1], 0.99)


if __name__ == '__main__':
    unittest.main()
