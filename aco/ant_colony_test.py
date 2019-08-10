import unittest
import pandas as pd

from aco.ant_colony import BinaryFeatureSelectionAntColony


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        frame = pd.read_csv('../data/Admission_Predict_Ver1.1.csv')
        self.featureA = frame.iloc[:, 1:2].to_numpy()
        self.featureB = frame.iloc[:, 2:3].to_numpy()
        self.featureC = frame.iloc[:, 3:4].to_numpy()
        self.featureD = frame.iloc[:, 4:5].to_numpy()
        self.featureE = frame.iloc[:, 5:6].to_numpy()
        self.featureF = frame.iloc[:, 6:7].to_numpy()
        self.featureG = frame.iloc[:, 7:8].to_numpy()
        self.featureH = frame.iloc[:, 8:9].to_numpy()
        self.result = frame.iloc[:, len(frame.columns) - 1].to_numpy()

    def test_ACO(self):
        data_set = [self.featureA, self.featureB, self.featureC, self.featureD, self.featureE, self.featureF,
                    self.featureG, self.featureH, self.result]
        ant_colony = BinaryFeatureSelectionAntColony(3, 4, 0.05, data_set, 2, 25)
        result_path = ant_colony.run()
        print(f"Result {result_path}")
        self.assertGreaterEqual(result_path[1], 0.99)


if __name__ == '__main__':
    unittest.main()
