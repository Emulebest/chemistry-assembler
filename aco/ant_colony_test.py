import unittest
from typing import List

import numpy as np

from aco.ant_colony import BinaryFeatureSelectionAntColony


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.featureA = np.array([[1], [2], [3], [4]])
        self.featureB = np.array([[100], [107], [112], [108]])
        self.result = np.array([[2], [4], [6], [8]])

    def test_ACO(self):
        construction_matrix: List[List[int]] = [[0, 0],
                                                [1, 1]]
        data_set = [self.featureA, self.featureB, self.result]
        ant_colony = BinaryFeatureSelectionAntColony(construction_matrix, 5, 5, 0.05, data_set, 1)
        result_path = ant_colony.run()
        self.assertEqual(result_path, ([(0, 1), (1, 0)], 1.0))


if __name__ == '__main__':
    unittest.main()
