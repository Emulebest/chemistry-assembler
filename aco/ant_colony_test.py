import unittest
from typing import List

import numpy as np

from aco.ant_colony import BinaryFeatureSelectionAntColony


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.featureA = np.random.randint(10000, size=(200, 1))
        self.featureB = np.random.randint(10000, size=(200, 1))
        self.featureC = np.random.randint(10000, size=(200, 1))
        self.featureD = np.random.randint(10000, size=(200, 1))
        self.featureE = np.random.randint(10000, size=(200, 1))
        self.featureF = np.random.randint(10000, size=(200, 1))
        self.result = np.random.randint(10000, size=(200, 1))

    def test_ACO(self):
        data_set = [self.featureA, self.featureB, self.featureC, self.featureD, self.featureE, self.featureF, self.result]
        ant_colony = BinaryFeatureSelectionAntColony(2, 4, 0.05, data_set, 2, 10)
        result_path = ant_colony.run()
        print(f"Result {result_path}")
        self.assertGreaterEqual(result_path[1], 0.99)


if __name__ == '__main__':
    unittest.main()
